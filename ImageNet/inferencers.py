import dataclasses
from retriever import DynamicReteiever
from tqdm import tqdm
import torch
from imagenet_dataset import ImageNetDataset
import os
from PIL import Image
from classification_utils import IMAGENET_CLASSNAMES_100
from torch.utils.data import Subset
from typing import List
from utils import get_topk_classifications
import numpy as np
import random

@dataclasses.dataclass
class Sample:
    idx: int
    image: Image
    label:str
    embed: torch.TensorType
    class_id:int
    pseudo_label: str or None
    pred_score: float or None


class Online_ICL:
    """
    Inference code for Online_ICL. You can inference your data with two steps:
    1). Init:             inferencer = Online_ICL(**kwargs)
    2). inference:        inferencer.run()
    """

    def __init__(self, args, tokenizer, model, image_processor,embedding_model, embedding_processor, device,processor=None):
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.embedding_model = embedding_model
        self.embedding_processor= embedding_processor
        self.device = device
        self.processor = processor
        self.test_sample_num = 0
        self.right_sample_num = 0
        self.all_class_names = IMAGENET_CLASSNAMES_100
        self.retriever = DynamicReteiever(args)
        self.predictions = []
        self.topk = 1

    def get_embedding(self, image):
        inputs = self.embedding_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.embedding_model.get_image_features(**inputs)
        return image_features

    def prepare_image(self,img_list):
        vision_x = [self.image_processor(img).unsqueeze(0) for img in img_list]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)
        vision_x = vision_x.to(self.device).half()
        return vision_x
    
    def prepare_text(self,ice_text):
        self.tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
        lang_x = self.tokenizer(ice_text, return_tensors="pt", )
        lang_x = {k: v.to(self.device) for k, v in lang_x.items()}
        return lang_x

    def get_response_OFv2(self, sample):
        ice_img,ice_text,demonstrations = self.retriever.get_final_query(sample)
        vision_x = self.prepare_image(ice_img)
        lang_x = self.prepare_text(ice_text)
        with torch.no_grad():
            outputs = self.model.generate(
                vision_x=vision_x,
                lang_x=lang_x["input_ids"],
                attention_mask=lang_x["attention_mask"],
                max_new_tokens=20,
                min_new_tokens=1,
                num_beams=1,
                output_scores=True,
                return_dict_in_generate=True
            )

        classnames_tokens = self.tokenizer(
            self.all_class_names
        )["input_ids"]

        predicted_classnames,predicted_logprobs = get_topk_classifications(outputs,classnames_tokens,self.topk)
        # compute accuracy
        y_i = sample.label

        self.predictions.append(
            {
                "id": sample.idx,
                "gt_label": y_i,
                "pred_label": predicted_classnames[0],
                "gt_id": sample.class_id,
                "pred_score": predicted_logprobs,
                "prompt_text":ice_text,
                "prompt_label":[dm.class_id for dm in demonstrations]
            }
        )
        sample.pred_score = predicted_logprobs
        sample.pseudo_label = predicted_classnames[0]
        self.retriever.update()

    def get_response_idefics(self,sample):
        demonstrations = self.retriever.get_demonstrations_from_bank(sample)
        prompt = []
        if demonstrations is not None:
            for dm in demonstrations:
                prompt.append(dm.image)
                prompt.append(f"Output:{dm.pseudo_label}"+"\n")
        prompt.append(sample.image)
        prompt.append(f"Output:")
        inputs = self.processor(prompt, return_tensors="pt").to(self.device)
        bad_words_ids = self.tokenizer(["<image>", "<fake_token_around_image>"],
                                            add_special_tokens=False).input_ids

        with torch.no_grad():
            outputs = self.model.generate(**inputs,
                                          max_new_tokens=20,
                                          bad_words_ids=bad_words_ids,
                                          output_scores=True,
                                        return_dict_in_generate=True)

        classnames_tokens = self.tokenizer(
            self.all_class_names
        )["input_ids"]
        print("classnames_tokens",classnames_tokens)
        predicted_classnames, predicted_logprobs, average_log_prob = get_topk_classifications(outputs,
                                                                                              classnames_tokens,
                                                                                              self.topk)
        # compute accuracy
        y_i = sample.label
        relative_score = torch.exp(torch.tensor(predicted_logprobs[0] - average_log_prob)).item()
        self.predictions.append(
            {
                "id": sample.idx,
                "gt_label": y_i,
                "pred_label": predicted_classnames[0],
                "gt_id": sample.class_id,
                "pred_score": relative_score,
                "prompt":prompt
            }
        )
        sample.pseudo_label = predicted_classnames[0]

    def inference(self, sample):
        sample = self.preprocess(sample)
        self.test_sample_num += 1
        if self.args.model == "open_flamingo":
            self.get_response_OFv2(sample)
        if self.args.model == "idefics":
            self.get_response_idefics(sample)
        if sample.pseudo_label == sample.label:
            self.right_sample_num += 1

    def preprocess(self, sample):
        idx = sample["id"]
        image = sample["image"]
        label = sample["class_name"]
        class_id = sample["class_id"]
        embed = self.get_embedding(image).squeeze().cpu()
        sample = Sample(idx, image, label, embed,class_id, None,None)
        return sample
    
    def process_dict(self,sample):
        if sample.label not in self.retriever.label2sample:
            self.retriever.label2sample[sample.pseudo_label] = [sample]
        else:
            self.retriever.label2sample[sample.pseudo_label].append(sample)

    def run(self):
        results = {"online": 0,"last":0}
        predictions = {"0":[],"1":[]}
        train_dataset = ImageNetDataset(os.path.join("/data/hyh/imagenet/data", "train"))
        test_dataset = ImageNetDataset(os.path.join("/data/hyh/imagenet/data", "val"))
        validate_set = Subset(test_dataset, list(range(5000))) #取前5000个样本作为validate set

        # 从train_dataset中取support set
        support_set = []
        sample_pool = []
        print("get memory bank and sample pool ...")
        if self.args.dataset_mode == "balanced":
            for i in self.all_class_names:
                support_set.extend(train_dataset.get_data_list_by_class(class_name=i)[0:self.args.M//100])
                sample_pool.extend(train_dataset.get_data_list_by_class(class_name=i)[5:105])
        else: # unbalanced
            # 设置样本数量的循环模式
            cycle_length = 10  # 每10类循环一次
            cycle_pattern = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]  # 总数为50张的分布

            # 遍历所有类别，按照循环模式分配样本到 support_set
            for i in range(len(self.all_class_names)):
                class_name = self.all_class_names[i]
                num_support_samples = cycle_pattern[i % cycle_length]  # 根据类索引取样本数量
                class_samples = train_dataset.get_data_list_by_class(class_name=class_name)
    
                # 确保有足够的样本来选择
                if len(class_samples) < num_support_samples + 110:
                    raise ValueError(f"Not enough samples for class {class_name}. Needed: {num_support_samples + 110}, Available: {len(class_samples)}")

                # 支持集样本
                support_set.extend(class_samples[:num_support_samples])
    
                # 样本池样本（从第10个到第110个，总共100个）
                sample_pool.extend(class_samples[10:110])

        # 输出支持集和样本池大小以确认正确性
        print(f"Support set size: {len(support_set)}, Sample pool size: {len(sample_pool)}")

        for idx in tqdm(range(len(support_set)), desc=f"Preprocess Supporting set..."):
            # 对sample进行预处理
            support_sample = self.preprocess(support_set[idx])
            self.retriever.demonstrations.append(support_sample)
            # 对 prototype 映射表 更新
            self.process_dict(support_sample)

        for idx in tqdm(range(len(sample_pool)), desc=f"Preprocess Sample Pool ..."):
            # 对sample进行预处理
            sample_pool_sample = self.preprocess(sample_pool[idx])
            self.retriever.pool.append(sample_pool_sample)

        # 设置随机种子
        random.seed(self.args.seed)
        # 打乱 validate_set 的索引
        shuffled_validate_indices = list(range(len(validate_set)))
        random.shuffle(shuffled_validate_indices)
        # 创建一个新的 Subset 作为打乱后的 validate_dataset
        shuffled_validate_dataset = Subset(test_dataset, shuffled_validate_indices)

        self.test_sample_num = 0
        self.right_sample_num = 0
        for idx in tqdm(range(len(shuffled_validate_dataset)), desc=f"Inference ImageNet..."):
            self.inference(shuffled_validate_dataset[idx])

        print(f"Successfully update the support set with sample pool, now support set size: {len(self.retriever.demonstrations)}, Sample pool size: {len(self.retriever.pool)}")

        acc = self.right_sample_num / self.test_sample_num
        results["online"] += acc
        predictions["0"] = self.predictions
        print("the online performance is:",results["online"])
        print("-------------------------------------------------------------")

        print("Inference using the latest supporting set...")

        self.predictions = []
        self.test_sample_num = 0
        self.right_sample_num = 0
        for idx in tqdm(range(len(shuffled_validate_dataset)), desc=f"Inference ImageNet..."):
            self.inference(shuffled_validate_dataset[idx])
        acc = self.right_sample_num / self.test_sample_num
        results["last"] += acc
        predictions["1"] = self.predictions

        return results,predictions
    
class FewShot:
    """
    Inference code for FewShot. You can inference your data with two steps:
    1). Init:             inferencer = FewShot(**kwargs)
    2). inference:        inferencer.run()
    """

    def __init__(self, args, tokenizer, model, image_processor,embedding_model, embedding_processor, device,processor=None):
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.embedding_model = embedding_model
        self.embedding_processor= embedding_processor
        self.device = device
        self.processor = processor
        self.test_sample_num = 0
        self.right_sample_num = 0
        self.all_class_names = IMAGENET_CLASSNAMES_100
        self.retriever = DynamicReteiever(args)
        self.predictions = []
        self.topk = 1

    def get_embedding(self, image):
        inputs = self.embedding_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.embedding_model.get_image_features(**inputs)
        return image_features

    def prepare_image(self,img_list):
        vision_x = [self.image_processor(img).unsqueeze(0) for img in img_list]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)
        vision_x = vision_x.to(self.device).half()
        return vision_x
    
    def prepare_text(self,ice_text):
        self.tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
        lang_x = self.tokenizer(ice_text, return_tensors="pt", )
        lang_x = {k: v.to(self.device) for k, v in lang_x.items()}
        return lang_x

    def get_response_OFv2(self, sample):
        ice_img,ice_text,demonstrations = self.retriever.get_final_query(sample)
        vision_x = self.prepare_image(ice_img)
        lang_x = self.prepare_text(ice_text)
        with torch.no_grad():
            outputs = self.model.generate(
                vision_x=vision_x,
                lang_x=lang_x["input_ids"],
                attention_mask=lang_x["attention_mask"],
                max_new_tokens=20,
                min_new_tokens=1,
                num_beams=1,
                output_scores=True,
                return_dict_in_generate=True
            )

        classnames_tokens = self.tokenizer(
            self.all_class_names
        )["input_ids"]

        predicted_classnames,predicted_logprobs = get_topk_classifications(outputs,classnames_tokens,self.topk)
        # compute accuracy
        y_i = sample.label

        self.predictions.append(
            {
                "id": sample.idx,
                "gt_label": y_i,
                "pred_label": predicted_classnames[0],
                "gt_id": sample.class_id,
                "pred_score": predicted_logprobs,
                "prompt_text":ice_text,
                "prompt_label":[dm.class_id for dm in demonstrations]
            }
        )
        sample.pred_score = predicted_logprobs
        sample.pseudo_label = predicted_classnames[0]

    def get_response_idefics(self,sample):
        demonstrations = self.retriever.get_demonstrations_from_bank(sample)
        prompt = []
        if demonstrations is not None:
            for dm in demonstrations:
                prompt.append(dm.image)
                prompt.append(f"Output:{dm.pseudo_label}"+"\n")
        prompt.append(sample.image)
        prompt.append(f"Output:")
        inputs = self.processor(prompt, return_tensors="pt").to(self.device)
        bad_words_ids = self.tokenizer(["<image>", "<fake_token_around_image>"],
                                            add_special_tokens=False).input_ids

        with torch.no_grad():
            outputs = self.model.generate(**inputs,
                                          max_new_tokens=20,
                                          bad_words_ids=bad_words_ids,
                                          output_scores=True,
                                        return_dict_in_generate=True)

        classnames_tokens = self.tokenizer(
            self.all_class_names
        )["input_ids"]
        print("classnames_tokens",classnames_tokens)
        predicted_classnames, predicted_logprobs, average_log_prob = get_topk_classifications(outputs,
                                                                                              classnames_tokens,
                                                                                              self.topk)
        # compute accuracy
        y_i = sample.label
        relative_score = torch.exp(torch.tensor(predicted_logprobs[0] - average_log_prob)).item()
        self.predictions.append(
            {
                "id": sample.idx,
                "gt_label": y_i,
                "pred_label": predicted_classnames[0],
                "gt_id": sample.class_id,
                "pred_score": relative_score,
                "prompt":prompt
            }
        )
        sample.pseudo_label = predicted_classnames[0]
        

    def inference(self, sample):
        sample = self.preprocess(sample)
        self.test_sample_num += 1
        if self.args.model == "open_flamingo":
            self.get_response_OFv2(sample)
        if self.args.model == "idefics":
            self.get_response_idefics(sample)
        if sample.pseudo_label == sample.label:
            self.right_sample_num += 1

    def preprocess(self, sample):
        idx = sample["id"]
        image = sample["image"]
        label = sample["class_name"]
        class_id = sample["class_id"]
        embed = self.get_embedding(image).squeeze().cpu()
        sample = Sample(idx, image, label, embed,class_id, label,None)
        return sample
    
    def run(self):
        results = {"avg": 0}
        train_dataset = ImageNetDataset(os.path.join("/data/hyh/imagenet/data", "train"))
        test_dataset = ImageNetDataset(os.path.join("/data/hyh/imagenet/data", "val"))
        validate_set = Subset(test_dataset, list(range(5000))) #取前5000个样本作为validate set

        # 从train_dataset中取support set
        support_set = []
        print("get supportng set ...")
        if self.args.dataset_mode == "balanced":
            # 取前100类对应的图片,每类取self.args.M//100张,也就是supporting set的大小是self.args.M
            for i in self.all_class_names:
                support_set.extend(train_dataset.get_data_list_by_class(class_name=i)[0:self.args.M//100])
        else: # unbalanced
            cycle_length = 10  # 每10类循环一次
            cycle_pattern = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]  # 总数为50张的分布
            for i in range(len(self.all_class_names)):
                class_name = self.all_class_names[i]
                num_support_samples = cycle_pattern[i % cycle_length]  # 根据类索引取样本数量
                class_samples = train_dataset.get_data_list_by_class(class_name=class_name)
    
                # 支持集样本
                support_set.extend(class_samples[:num_support_samples])

        # 输出支持集 大小以确认正确性
        print(f"Support set size: {len(support_set)}")

        for idx in tqdm(range(len(support_set)), desc=f"Preprocess Supporting set..."):
            # 对sample进行预处理
            support_sample = self.preprocess(support_set[idx])
            self.retriever.demonstrations.append(support_sample)

        # 设置随机种子
        random.seed(self.args.seed)
        # 打乱 validate_set 的索引
        shuffled_validate_indices = list(range(len(validate_set)))
        random.shuffle(shuffled_validate_indices)
        # 创建一个新的 Subset 作为打乱后的 validate_dataset
        shuffled_validate_dataset = Subset(test_dataset, shuffled_validate_indices)

        self.test_sample_num = 0
        self.right_sample_num = 0
        for idx in tqdm(range(len(shuffled_validate_dataset)), desc=f"Inference ImageNet..."):
            self.inference(shuffled_validate_dataset[idx])
        acc = self.right_sample_num / self.test_sample_num
        results["avg"] += acc

        return results,self.predictions
