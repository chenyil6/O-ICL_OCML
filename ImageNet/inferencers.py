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
    value: int


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
        sample = Sample(idx, image, label, embed,class_id, None,None,0)
        return sample
    
    def process_dict(self,sample):
        if sample.label not in self.retriever.label2sample:
            self.retriever.label2sample[sample.label] = [sample]
        else:
            self.retriever.label2sample[sample.label].append(sample)

    def run(self):
        results = {"avg":0}
        train_dataset = ImageNetDataset(os.path.join("/data/hyh/imagenet/data", "train"))
        test_dataset = ImageNetDataset(os.path.join("/data/hyh/imagenet/data", "val"))
        validate_set = Subset(test_dataset, list(range(5000))) #取前5000个样本作为validate set
        # 设置全局随机种子
        random.seed(self.args.seed)
    
        validate_rng = random.Random(self.args.seed)  # 用于打乱 validate_set 的随机生成器
        sample_pool_rng = random.Random(self.args.seed + 1)  # 用于打乱 sample_pool 的随机生成器
        class_selection_rng = random.Random(self.args.seed + 3)  # 用于选择 no_sample_classes 的随机生成器
    
        # 从train_dataset中取support set
        support_set = []
        sample_pool = []
        print("get memory bank and sample pool ...")
        if self.args.dataset_mode == "balanced":
            for i in range(len(self.all_class_names)):
                class_name = self.all_class_names[i]
                data_list = train_dataset.get_data_list_by_class(class_name=class_name)
                support_set.extend(data_list[0:self.args.M//100])
                sample_pool.extend(data_list[100:200])
        else: # imbalanced
            num_classes = len(self.all_class_names)
            no_sample_classes = class_selection_rng.sample(range(num_classes), num_classes // 2)

            for i in range(len(self.all_class_names)):
                class_name = self.all_class_names[i]
                class_samples = train_dataset.get_data_list_by_class(class_name=class_name)
                if i in no_sample_classes:
                    continue
                else:
                    support_set.extend(class_samples[:self.args.M*2//100])

                # 样本池样本（从第10个到第110个，总共100个）
                sample_pool.extend(class_samples[100:200])

        print(f"Support set size: {len(support_set)}, Sample pool size: {len(sample_pool)}")

        for idx in tqdm(range(len(support_set)), desc=f"Preprocess Supporting set..."):
            # 对sample进行预处理
            support_sample = self.preprocess(support_set[idx])
            self.retriever.demonstrations.append(support_sample)
            # 对 prototype 映射表 更新
            self.process_dict(support_sample)

        print(f"Get the value of every sample in support set")
        if self.args.update_strategy == "SV_weight":
            all_embeds = [s.embed for s in self.retriever.demonstrations]
            for sample in self.retriever.demonstrations:
                query_embed = sample.embed
                # 计算 L2 距离
                distances = torch.norm(torch.stack(all_embeds) - query_embed.unsqueeze(0), dim=1)
                weights = 1 / (distances + 1e-8)
                similarities = torch.cosine_similarity(torch.stack(all_embeds),query_embed.unsqueeze(0))
                _, top_indices = torch.topk(similarities, k=11)  

                x1 = sum(weights[idx].item() for idx in top_indices[1:] if self.retriever.demonstrations[idx].label == sample.label)  
                x2 = sum(weights[idx].item() for idx in top_indices[1:] if self.retriever.demonstrations[idx].label != sample.label)  
                sample.value = x1-x2 
        if self.args.update_strategy == "SV":
            all_embeds = [s.embed for s in self.retriever.demonstrations]
            for sample in self.retriever.demonstrations:
                query_embed = sample.embed
                similarities = torch.cosine_similarity(torch.stack(all_embeds),query_embed.unsqueeze(0))
                _, top_indices = torch.topk(similarities, k=11)  # 获取前11个相似样本的索引

                x1 = sum(1 for idx in top_indices[1:] if self.retriever.demonstrations[idx].label == sample.label)  
                x2 = sum(1 for idx in top_indices[1:] if self.retriever.demonstrations[idx].label != sample.label)  
                sample.value = x1-x2 
        
        if self.args.update_strategy == "value":
            all_embeds = [s.embed for s in self.retriever.demonstrations]
            for sample in self.retriever.demonstrations:
                query_embed = sample.embed  
                similarities = torch.cosine_similarity(torch.stack(all_embeds),query_embed.unsqueeze(0))
                _, top_indices = torch.topk(similarities, k=11)  # 获取前11个相似样本的索引
                x1 = sum(1 for idx in top_indices[1:] if self.retriever.demonstrations[idx].label == sample.label)  
                x2 = sum(1 for idx in top_indices[1:] if self.retriever.demonstrations[idx].label != sample.label)  
                label_consistency = (x1-x2)/10.0

                # 计算对应类的prototype
                label = sample.label
                sample_list = self.retriever.label2sample[label]
                embed_list = [sample.embed for sample in sample_list]
                prototype = torch.mean(torch.stack(embed_list), dim=0)
                proto_similarity = torch.cosine_similarity(query_embed.unsqueeze(0), prototype.unsqueeze(0)).item()
                sample.value = self.args.alpha * label_consistency + (1 - self.args.alpha) * proto_similarity

        sample_pool_rng.shuffle(sample_pool)  # 使用单独的 random 对象打乱 sample_pool

        for idx in tqdm(range(len(sample_pool)), desc=f"Preprocess Sample Pool ..."):
            # 对sample进行预处理
            sample_pool_sample = self.preprocess(sample_pool[idx])
            self.retriever.pool.append(sample_pool_sample)

        # 打乱 validate_set 的索引
        shuffled_validate_indices = list(range(len(validate_set)))
        validate_rng.shuffle(shuffled_validate_indices)
        # 创建一个新的 Subset 作为打乱后的 validate_dataset
        shuffled_validate_dataset = Subset(test_dataset, shuffled_validate_indices)

        self.test_sample_num = 0
        self.right_sample_num = 0

        for idx in tqdm(range(len(sample_pool)), desc=f"Update the support set with sample pool..."):
            self.retriever.update()

        print(f"Successfully update the support set with sample pool, now support set size: {len(self.retriever.demonstrations)}, Sample pool size: {len(self.retriever.pool)}")

        print("Inference using the latest supporting set...")

        self.test_sample_num = 0
        self.right_sample_num = 0
        for idx in tqdm(range(len(shuffled_validate_dataset)), desc=f"Inference ImageNet..."):
            self.inference(shuffled_validate_dataset[idx])
        acc = self.right_sample_num / self.test_sample_num
        results["avg"] += acc

        return results,self.predictions
    
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
        # 设置全局随机种子
        random.seed(self.args.seed)
    
        # 创建不同的随机数生成器
        validate_rng = random.Random(self.args.seed)
        class_selection_rng = random.Random(self.args.seed + 3)
        # 从train_dataset中取support set
        support_set = []
        print("get supportng set ...")
        if self.args.dataset_mode == "balanced":
            # 取前100类对应的图片,每类取self.args.M//100张,也就是supporting set的大小是self.args.M
            for i in self.all_class_names:
                support_set.extend(train_dataset.get_data_list_by_class(class_name=i)[0:self.args.M//100])
        else: # unbalanced
            num_classes = len(self.all_class_names)
            no_sample_classes = class_selection_rng.sample(range(num_classes), num_classes // 2)
            for i in range(len(self.all_class_names)):
                class_name = self.all_class_names[i]
                class_samples = train_dataset.get_data_list_by_class(class_name=class_name)
                if i in no_sample_classes:
                    continue
                else:
                    support_set.extend(class_samples[:self.args.M*2//100])
                    
        # 输出支持集 大小以确认正确性
        print(f"Support set size: {len(support_set)}")

        for idx in tqdm(range(len(support_set)), desc=f"Preprocess Supporting set..."):
            # 对sample进行预处理
            support_sample = self.preprocess(support_set[idx])
            self.retriever.demonstrations.append(support_sample)

        # 打乱 validate_set 的索引
        shuffled_validate_indices = list(range(len(validate_set)))
        validate_rng.shuffle(shuffled_validate_indices)
        # 创建一个新的 Subset 作为打乱后的 validate_dataset
        shuffled_validate_dataset = Subset(test_dataset, shuffled_validate_indices)

        self.test_sample_num = 0
        self.right_sample_num = 0
        for idx in tqdm(range(len(shuffled_validate_dataset)), desc=f"Inference ImageNet..."):
            self.inference(shuffled_validate_dataset[idx])
        acc = self.right_sample_num / self.test_sample_num
        results["avg"] += acc

        return results,self.predictions

class Online_ICL_New:
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
        self.count = dict()

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

        predicted_classnames,predicted_logprobs = get_topk_classifications(outputs,classnames_tokens,1)
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
                                                                                              1)
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
        # 不管预测结果如何，都要传入两个参数label,isCorrect
        self.retriever.update_new(sample.label,sample.pseudo_label == sample.label,self.args.prob)

    def preprocess(self, sample):
        idx = sample["id"]
        image = sample["image"]
        label = sample["class_name"]
        class_id = sample["class_id"]
        embed = self.get_embedding(image).squeeze().cpu()
        sample = Sample(idx, image, label, embed,class_id, None,None)
        return sample
    
    def process_dict(self,sample):
        # 不仅对memory bank的 prototype 映射表更新, 还要对 retriever 的 self.accept_prob 和 self.drop_prob 更新
        if sample.label not in self.retriever.label2sample:
            self.retriever.label2sample[sample.label] = [sample]
        else:
            self.retriever.label2sample[sample.label].append(sample)
        if sample.label not in self.retriever.probs:
            self.retriever.probs[sample.label]  = 1.0
        if sample.label not in self.retriever.count:
            self.retriever.count[sample.label] =0
        self.retriever.count[sample.label] +=1
        if sample.label not in self.retriever.all_meet:
            self.retriever.all_meet[sample.label] =0
        self.retriever.all_meet[sample.label] +=1
        #if sample.label not in self.retriever.accept_prob:
            #self.retriever.accept_prob[sample.label]  = 1.0
        #if sample.label not in self.retriever.drop_prob:
            #self.retriever.drop_prob[sample.label]  = 0.0

    def run(self):
        results = {"avg": 0,}
        train_dataset = ImageNetDataset(os.path.join("/data/hyh/imagenet/data", "train"))
        test_dataset = ImageNetDataset(os.path.join("/data/hyh/imagenet/data", "val"))
        test_set = Subset(test_dataset, list(range(5000))) #取前5000个样本作为validate set
        # 设置全局随机种子
        random.seed(self.args.seed)
    
        # 创建不同的随机数生成器
        test_rng = random.Random(self.args.seed)  # 用于打乱 test_set 的随机生成器
        sample_pool_rng = random.Random(self.args.seed + 1)  
        val_rng = random.Random(self.args.seed + 2) 
        class_selection_rng = random.Random(self.args.seed + 3)
        # 从train_dataset中取 support_set (500个）和 val_set(2000个) sample pool（10000个）
        support_set = []
        val_set = []
        sample_pool = []

        print("get memory bank and sample pool and val_set...")

        if self.args.dataset_mode == "balanced":
            for i in self.all_class_names:
                data_list = train_dataset.get_data_list_by_class(class_name=i)
                support_set.extend(data_list[0:self.args.M//100])
                sample_pool.extend(data_list[100:200])
                val_set.extend(data_list[300:320])
        else:
            num_classes = len(self.all_class_names)
            no_sample_classes = class_selection_rng.sample(range(num_classes), num_classes // 2)
            for i in range(num_classes):
                class_name = self.all_class_names[i]
                class_samples = train_dataset.get_data_list_by_class(class_name=class_name)
                if i in no_sample_classes:
                    continue
                else:
                    support_set.extend(class_samples[:self.args.M*2//100])
                sample_pool.extend(data_list[100:200])
                val_set.extend(data_list[300:320])

        print(f"Support set size: {len(support_set)}, Sample pool size: {len(sample_pool)},Val set size:{len(val_set)}")

        for idx in tqdm(range(len(support_set)), desc=f"Preprocess Supporting set..."):
            support_set_sample = self.preprocess(support_set[idx])
            self.process_dict(support_set_sample)
            self.retriever.demonstrations.append(support_set_sample)
    
        sample_pool_rng.shuffle(sample_pool)  # 使用单独的 random 对象打乱 sample_pool
        print("preprocess the samples in sample pool ...")
        for idx in tqdm(range(len(sample_pool)), desc=f"Preprocess Sample Pool..."):
            sample_pool_sample = self.preprocess(sample_pool[idx])
            self.retriever.pool.append(sample_pool_sample)

        print(f"support set initial size={len(self.retriever.demonstrations)}, and sample pool initial size = {len(self.retriever.pool)}")

        # 打乱 test_set 的索引
        shuffled_test_indices = list(range(len(test_set)))
        test_rng.shuffle(shuffled_test_indices)
        # 创建一个新的 Subset 作为打乱后的 test_set
        test_set = Subset(test_dataset, shuffled_test_indices)

        # 打乱 val_set 的索引
        val_rng.shuffle(val_set)  # 使用随机数生成器 val_rng 对 val_set 进行打乱
        for idx in tqdm(range(len(val_set)), desc=f"Inference val set..."):
            self.inference(val_set[idx])

        self.predictions = []

        # 当 验证集推理完成后，sample pool 为空， memory bank更新完毕，就用最新的memory bank 再来对测试集做评估
        print(f"support set size={len(self.retriever.demonstrations)}, and sample pool  size = {len(self.retriever.pool)}")
        print("inference with updated memory bank...")
        self.test_sample_num = 0
        self.right_sample_num = 0
        for idx in tqdm(range(len(test_set)), desc=f"Inference ImageNet..."):
            self.inference(test_set[idx])
        
        acc = self.right_sample_num / self.test_sample_num
        results["avg"] += acc
        return results,self.predictions

class Online_ICL_Count: # 2000个样本纯粹是推理的，用来计算正确率
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
        self.count = dict()
        self.class_acc = dict()

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

        predicted_classnames,predicted_logprobs = get_topk_classifications(outputs,classnames_tokens,1)
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
                                                                                              1)
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
        self.count[sample.label][1] += 1
        if self.args.model == "open_flamingo":
            self.get_response_OFv2(sample)
        if self.args.model == "idefics":
            self.get_response_idefics(sample)
        if sample.pseudo_label == sample.label:
            self.right_sample_num += 1
            self.count[sample.label][0] += 1
            
    def preprocess(self, sample):
        idx = sample["id"]
        image = sample["image"]
        label = sample["class_name"]
        class_id = sample["class_id"]
        embed = self.get_embedding(image).squeeze().cpu()
        sample = Sample(idx, image, label, embed,class_id, None,None)
        return sample
    
    def process_dict(self,sample):
        # 不仅对memory bank的 prototype 映射表更新, 还要对 retriever 的 self.accept_prob 和 self.drop_prob 更新
        if sample.label not in self.retriever.label2sample:
            self.retriever.label2sample[sample.label] = [sample]
        else:
            self.retriever.label2sample[sample.label].append(sample)
        if sample.label not in self.count:
            self.count[sample.label] =[0,0]
        

    def run(self):
        results = {"avg": 0,}
        train_dataset = ImageNetDataset(os.path.join("/data/hyh/imagenet/data", "train"))
        test_dataset = ImageNetDataset(os.path.join("/data/hyh/imagenet/data", "val"))
        test_set = Subset(test_dataset, list(range(5000))) #取前5000个样本作为validate set
        # 设置全局随机种子
        random.seed(self.args.seed)
    
        # 创建不同的随机数生成器
        test_rng = random.Random(self.args.seed)  # 用于打乱 test_set 的随机生成器
        sample_pool_rng = random.Random(self.args.seed + 1)  
        val_rng = random.Random(self.args.seed + 2) 
        class_selection_rng = random.Random(self.args.seed + 3)
        # 从train_dataset中取 support_set (500个）和 val_set(2000个) sample pool（10000个）
        support_set = []
        val_set = []
        sample_pool = []

        print("get memory bank and sample pool and val_set...")

        if self.args.dataset_mode == "balanced":
            for i in self.all_class_names:
                data_list = train_dataset.get_data_list_by_class(class_name=i)
                support_set.extend(data_list[0:self.args.M//100])
                sample_pool.extend(data_list[100:200])
                val_set.extend(data_list[300:320])
        else:
            num_classes = len(self.all_class_names)
            no_sample_classes = class_selection_rng.sample(range(num_classes), num_classes // 2)
            for i in range(num_classes):
                class_name = self.all_class_names[i]
                class_samples = train_dataset.get_data_list_by_class(class_name=class_name)
                if i in no_sample_classes:
                    continue
                else:
                    support_set.extend(class_samples[:self.args.M*2//100])
                sample_pool.extend(data_list[100:200])
                val_set.extend(data_list[300:320])

        print(f"Support set size: {len(support_set)}, Sample pool size: {len(sample_pool)},Val set size:{len(val_set)}")

        for idx in tqdm(range(len(support_set)), desc=f"Preprocess Supporting set..."):
            support_set_sample = self.preprocess(support_set[idx])
            self.process_dict(support_set_sample)
            self.retriever.demonstrations.append(support_set_sample)
    
        sample_pool_rng.shuffle(sample_pool)  # 使用单独的 random 对象打乱 sample_pool
        print("preprocess the samples in sample pool ...")
        for idx in tqdm(range(len(sample_pool)), desc=f"Preprocess Sample Pool..."):
            sample_pool_sample = self.preprocess(sample_pool[idx])
            self.retriever.pool.append(sample_pool_sample)

        print(f"support set initial size={len(self.retriever.demonstrations)}, and sample pool initial size = {len(self.retriever.pool)}")

        # 打乱 test_set 的索引
        shuffled_test_indices = list(range(len(test_set)))
        test_rng.shuffle(shuffled_test_indices)
        # 创建一个新的 Subset 作为打乱后的 test_set
        test_set = Subset(test_dataset, shuffled_test_indices)

        # 打乱 val_set 的索引
        val_rng.shuffle(val_set)  
        for idx in tqdm(range(len(val_set)), desc=f"Inference val set..."):
            self.inference(val_set[idx])

        # 推理完之后，可以得到每个类的accuracy
        for c,[correct,all] in self.count.items():
            self.class_acc[c] = float(correct)/all

        # 再根据每个类的 accuracy 决定如何更新 support set
        for idx in tqdm(range(len(sample_pool)), desc=f"Update the support set with sample pool..."):
            self.retriever.update_count(self.class_acc)

        self.predictions = []

        # 当 验证集推理完成后，sample pool 为空， memory bank更新完毕，就用最新的memory bank 再来对测试集做评估
        print(f"support set size={len(self.retriever.demonstrations)}, and sample pool  size = {len(self.retriever.pool)}")
        print("inference with updated memory bank...")
        self.test_sample_num = 0
        self.right_sample_num = 0
        for idx in tqdm(range(len(test_set)), desc=f"Inference ImageNet..."):
            self.inference(test_set[idx])
        
        acc = self.right_sample_num / self.test_sample_num
        results["avg"] += acc
        return results,self.predictions
