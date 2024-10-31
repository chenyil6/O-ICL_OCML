import dataclasses
import retriever

import importlib
# 重新加载 retriever 模块以应用最新的更改
importlib.reload(retriever)
from retriever import DynamicReteiever
from tqdm import tqdm
import torch
from imagenet_dataset import ImageNetDataset
import os
from PIL import Image
from classification_utils import IMAGENET_CLASSNAMES_100,IMAGENET_CLASSNAMES
from torch.utils.data import Subset
from typing import Optional
from utils import get_topk_classifications,get_imagenet_prompt,get_topk_classifications_batch
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import json
import pickle
import logging
from typing import List


logger = logging.getLogger(__name__)

@dataclasses.dataclass
class Sample:
    idx: int
    image: Image.Image
    label: str
    embed: torch.Tensor
    quality: Optional[float]
    class_id: int
    pseudo_label: Optional[str]
    pred_score: Optional[float] 
    gt_score: Optional[float]     
    margin: Optional[float]      


class Online_ICL:
    """
    Inference code for Online_ICL. You can inference your data with two steps:
    1). Init:             inferencer = Online_ICL(**kwargs)
    2). inference:        inferencer.run()
    """

    def __init__(self, args, tokenizer, model, image_processor,embedding_model, embedding_processor, embedding_tokenizer,device,processor=None):
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.embedding_model = embedding_model
        self.embedding_processor= embedding_processor
        self.embedding_tokenizer = embedding_tokenizer
        self.device = device
        self.processor = processor
        self.test_sample_num = 0
        self.right_sample_num = 0
        self.all_class_names = IMAGENET_CLASSNAMES_100
        self.retriever = DynamicReteiever(args)
        self.predictions = []
        self.topk = 1
        self.features_data_train = pickle.load(open("/data/chy/feacture_cache/train_idx2embed_quality.pkl", 'rb'))
        self.features_data_val = pickle.load(open("/data/chy/feacture_cache/val_idx2embed.pkl",'rb'))
        
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

    def get_text_embedding(self, label):
        inputs = self.embedding_tokenizer(text=label, padding=True,return_tensors="pt")
        with torch.no_grad():
            text_features = self.embedding_model.get_text_features(**inputs)
        return text_features
    
    def prepare_text(self,ice_text):
        self.tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
        lang_x = self.tokenizer(ice_text, return_tensors="pt", )
        lang_x = {k: v.to(self.device) for k, v in lang_x.items()}
        return lang_x

    def get_response_OFv2(self, sample):
        ice_img,ice_text,demonstrations = self.retriever.get_final_query(sample)

        # 把demonstration分别放入两个pool
        for dm in demonstrations:
            if dm.label == sample.label:
                self.retriever.match_pool.append(dm)
            else:
                self.retriever.dismatch_pool.append(dm)

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

        predicted_classnames,predicted_logprobs,overall_log_probs = get_topk_classifications(outputs,classnames_tokens,topk=2)
        # compute accuracy
        y_i = sample.label

        # Find the index of the ground truth label
        gt_label_index = self.all_class_names.index(y_i)

        # Get the confidence score of the ground truth label
        gt_label_confidence = overall_log_probs[gt_label_index].item()

        # margin M:top1的预测概率-top2的预测概率，得到的是模型对预测的不准确性。M越小，就越不稳定
        margin = predicted_logprobs[0] -predicted_logprobs[1]

        self.predictions.append(
            {
                "id": sample.idx,
                "gt_label": y_i,
                "pred_label": predicted_classnames[0],
                "gt_id": sample.class_id,
                "pred_score": predicted_logprobs[0],
                "gt_score": gt_label_confidence, 
                "margin":margin,
                "prompt_text":ice_text,
                "prompt_label":[dm.label for dm in demonstrations]
            }
        )
        sample.pred_score = predicted_logprobs[0]
        sample.pseudo_label = predicted_classnames[0]
        sample.gt_score = gt_label_confidence 
        sample.margin = margin
    
    def get_response_ideficsv2(self,sample):
        demonstrations = self.retriever.get_demonstrations_from_bank(sample)
        images = []
        prompts = []
        prompt = ""
        if demonstrations is not None:
            for dm in demonstrations:
                images.append(dm.image)
                prompt += f"Category:{dm.label}."
        images.append(sample.image)
        prompt += f"Category:"
        prompts.append(prompt)

        BAD_WORDS_IDS = self.processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
        EOS_WORDS_IDS = [self.processor.tokenizer.eos_token_id]
        
        inputs = self.processor(images=images, text=prompts, padding=True, truncation=True, return_tensors="pt").to("cuda")
       
        with torch.no_grad():
            outputs = self.model.generate(**inputs, bad_words_ids=BAD_WORDS_IDS,min_new_tokens=1, max_new_tokens=20,num_beams=1,
                        length_penalty=1.0,
                        output_scores=True,
                        return_dict_in_generate=True)

        classnames_tokens = self.tokenizer(
            self.all_class_names
        )["input_ids"]

        predicted_classnames,predicted_logprobs,overall_log_probs = get_topk_classifications(outputs,classnames_tokens,topk=2)
        # compute accuracy
        y_i = sample.label
        # Find the index of the ground truth label
        gt_label_index = self.all_class_names.index(y_i)

        # Get the confidence score of the ground truth label
        gt_label_confidence = overall_log_probs[gt_label_index].item()

        # margin M:top1的预测概率-top2的预测概率，得到的是模型对预测的不准确性。M越小，就越不稳定
        margin = predicted_logprobs[0] -predicted_logprobs[1]

        self.predictions.append(
            {
                "id": sample.idx,
                "gt_label": y_i,
                "pred_label": predicted_classnames[0],
                "gt_id": sample.class_id,
                "pred_score": predicted_logprobs[0],
                "gt_score": gt_label_confidence, 
                "margin":margin,
            }
        )
        sample.pred_score = predicted_logprobs[0]
        sample.pseudo_label = predicted_classnames[0]
        sample.gt_score = gt_label_confidence 
        sample.margin = margin

    def inference(self, sample):
        sample = self.preprocess_train(sample)
        self.test_sample_num += 1
        if self.args.model == "open_flamingo":
            self.get_response_OFv2(sample)
        if self.args.model == "idefics_v2":
            self.get_response_ideficsv1(sample)
        if sample.pseudo_label == sample.label:
            self.right_sample_num += 1
        self.retriever.update_online(sample)
        
    def evaluate_batch_on_OFv2(self, batch_samples):
        batch_images = []
        batch_text = []
        batch_demonstrations = []
        for sample in batch_samples: # 遍历每个sample，找到分别对应的context
            ice_img,ice_text,demonstrations = self.retriever.get_final_query(sample)
            batch_images.append(ice_img)
            batch_text.append(ice_text)
            batch_demonstrations.append(demonstrations)

        batch_images = self._prepare_images(batch_images)
        ctx_input_ids, ctx_attention_mask = self._prepare_text(batch_text)

        _lang_x = torch.cat([ctx_input_ids], dim=1)
        _attention_mask = torch.cat(
            [
                ctx_attention_mask,
            ],
            dim=1,
        )
        _vision_x = batch_images
        # 准备文本输入
        ctx_input_ids, ctx_attention_mask = self._prepare_text(batch_text)

        _lang_x = torch.cat([ctx_input_ids], dim=1)
        _attention_mask = torch.cat(
            [
                ctx_attention_mask,
            ],
            dim=1,
        )

        with torch.no_grad():
            outputs = self.model.generate(
                vision_x=_vision_x, 
                lang_x=_lang_x,
                attention_mask=_attention_mask,
                vision_features=None,
                max_new_tokens=20,
                min_new_tokens=1,
                num_beams=1,
                output_scores=True,
                return_dict_in_generate=True
            )

        # 后续处理
        classnames_tokens = self.tokenizer(self.all_class_names)["input_ids"]

        predicted_classnames_batch, predicted_logprobs_batch, overall_log_probs = get_topk_classifications_batch(
            outputs,
            classnames_tokens,
            self.topk,
        )

        # Process predictions for each sample
        for idx, sample in enumerate(batch_samples):
            y_i = sample.label
            gt_label_index = self.all_class_names.index(y_i)
            gt_label_confidence = overall_log_probs[idx, gt_label_index].item()
            demonstrations = batch_demonstrations[idx]
            self.predictions.append(
                {
                    "id": sample.idx,
                    "gt_label": y_i,
                    "pred_label": predicted_classnames_batch[idx][0],
                    "gt_id": sample.class_id,
                    "pred_score": predicted_logprobs_batch[idx][0],
                    "gt_score": gt_label_confidence,
                    "prompt_text": batch_text[idx],
                    "prompt_label": [dm.label for dm in demonstrations]
                }
            )
            sample.pred_score = predicted_logprobs_batch[idx][0]
            sample.pseudo_label = predicted_classnames_batch[idx][0]
            sample.gt_score = gt_label_confidence


    def _prepare_images(self, batch: List[List[Image.Image]]) -> torch.Tensor:
        """
        Convert images to tensors, reshape them, and stack them.
        Args:
            batch: A list of lists of images.
        Returns:
            preprocessed images (tensors) or None
                shape (B, T_img, F, C, H, W)
                None if no images in batch
        """
        images_per_example = max(len(x) for x in batch) # 一个批量有多少图片
        batch_images = None
        for iexample, example in enumerate(batch): 
            for iimage, image in enumerate(example):
                preprocessed = self.image_processor(image)
                if batch_images is None:
                    batch_images = torch.zeros(
                        (len(batch), images_per_example, 1) + preprocessed.shape,
                        dtype=preprocessed.dtype,
                    )
                batch_images[iexample, iimage, 0] = preprocessed
        if batch_images is not None:
            batch_images = batch_images.to(self.device).half()
        return batch_images

    def _prepare_text(
            self,
            batch: List[List[str]],
            padding="longest",
            truncation=True,
            max_length=2000,
        ):
            """
            Tokenize the text and stack them.
            Args:
                batch: A list of lists of strings.
            Returns:
                input_ids (tensor)
                    shape (B, T_txt)
                attention_mask (tensor)
                    shape (B, T_txt)
            """
            self.tokenizer.padding_side = "left"
            encodings = self.tokenizer(
                batch,
                padding=padding,
                truncation=truncation,
                return_tensors="pt",
                max_length=max_length,
            )
            input_ids, attention_mask = encodings["input_ids"], encodings["attention_mask"]
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            return input_ids, attention_mask

    def evaluate_batch_on_idev2(self,batch_samples):
        # Prepare prompts and image sets for batch processing
        prompts = []
        images = []
        
        for sample in batch_samples:
            demonstrations = self.retriever.get_demonstrations_from_bank(sample)
            prompt = ""
            image_set = []
            
            # Append demonstration images and labels
            if demonstrations is not None:
                for dm in demonstrations:
                    image_set.append(dm.image)
                    prompt += f"Category:{dm.label}."
            
            # Append the main sample image and category prompt
            image_set.append(sample.image)
            prompt += "Category:"
            
            # Collect prompts and images
            prompts.append(prompt)
            images.append(image_set)

        BAD_WORDS_IDS = self.processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
        EOS_WORDS_IDS = [self.processor.tokenizer.eos_token_id]
        
        inputs = self.processor(images=images, text=prompts, padding=True, truncation=True, return_tensors="pt").to("cuda")
       
        with torch.no_grad():
            outputs = self.model.generate(**inputs, bad_words_ids=BAD_WORDS_IDS,min_new_tokens=1, max_new_tokens=20,num_beams=1,
                        length_penalty=1.0,
                        output_scores=True,
                        return_dict_in_generate=True)

        classnames_tokens = self.tokenizer(self.all_class_names)["input_ids"]
        # Get top-k classifications for the batch
        predicted_classnames_batch, predicted_logprobs_batch, overall_log_probs = get_topk_classifications_batch(
            outputs,
            classnames_tokens,
            self.topk,
        )
        # Process predictions for each sample
        for idx, sample in enumerate(batch_samples):
            y_i = sample.label
            gt_label_index = self.all_class_names.index(y_i)
            gt_label_confidence = overall_log_probs[idx, gt_label_index].item()

            self.predictions.append(
                {
                    "id": sample.idx,
                    "gt_label": y_i,
                    "pred_label": predicted_classnames_batch[idx][0],
                    "gt_id": sample.class_id,
                    "pred_score": predicted_logprobs_batch[idx][0],
                    "gt_score": gt_label_confidence,
                    "prompt_label": [dm.class_id for dm in demonstrations]
                }
            )
            sample.pred_score = predicted_logprobs_batch[idx][0]
            sample.pseudo_label = predicted_classnames_batch[idx][0]
            sample.gt_score = gt_label_confidence

    def inference_batch(self,batch_samples):
        batch_samples = [self.preprocess_val(sample) for sample in batch_samples]
        self.test_sample_num += len(batch_samples)
        if self.args.model == "open_flamingo_9b":
            self.evaluate_batch_on_OFv2(batch_samples)
        if self.args.model == "idefics_v2":
            self.evaluate_batch_on_idev2(batch_samples)
        for sample in batch_samples:
            if sample.pseudo_label == sample.label:
                self.right_sample_num += 1

    def preprocess_train(self, sample):
        idx = sample["id"]  
        image = sample["image"]
        label = sample["class_name"]
        class_id = sample["class_id"]
        feature_64_1024 = self.features_data_train_64_1024[idx]
        feature_256_1024 = self.features_data_train_256_1024[idx]
        embed, quality = self.features_data_train[idx]
        sample = Sample(idx, image, label, embed,feature_64_1024,feature_256_1024,quality,class_id, None,None,None,None)
        return sample
    
    def preprocess_val(self, sample):
        idx = sample["id"]
        image = sample["image"]
        label = sample["class_name"]
        class_id = sample["class_id"]
        feature_64_1024 = self.features_data_val_64_1024[idx]
        feature_256_1024 = self.features_data_val_256_1024[idx]
        embed= self.features_data_val[idx]
        sample = Sample(idx, image, label, embed,feature_64_1024,feature_256_1024,None,class_id, None,None,None,None)
        return sample
    
    def process_dict(self,sample):
        if sample.label not in self.retriever.label2sample:
            self.retriever.label2sample[sample.label] = [sample]
        else:
            self.retriever.label2sample[sample.label].append(sample)

    def _initialize_prototypes(self):
        default_prototype_shape = (1, 768)
        default_zero_prototype = torch.zeros(default_prototype_shape)
        
        for label, samples in self.retriever.label2sample.items():
            if samples:
                embeddings = torch.stack([s.embed for s in samples])
                prototype = torch.mean(embeddings, dim=0)
                self.retriever.label_to_prototype[label] = prototype
            else:
                # 如果没有样本，使用默认的零张量
                self.retriever.label_to_prototype[label] = default_zero_prototype.clone()

    def visualize_tsne(self,title, save_path,perplexity=30, learning_rate=200):
        """
        对 memory bank 中的样本进行 t-SNE 可视化并保存为 JPG 文件
        """
        visualize_classes = ["electric ray", "coucal", "brambling", "southern black widow", 
                     "lorikeet", "newt", "rooster", "American dipper"]
        demonstrations = self.retriever.demonstrations
        # 提取属于这8个类别的样本
        selected_samples = [sample for sample in demonstrations if sample.label in visualize_classes]

        if len(selected_samples) == 0:
            print(f"No samples found for the selected classes: {visualize_classes}")
            return
    
        # 提取样本的 embed 和 label 信息
        embeds = torch.stack([sample.embed for sample in selected_samples]).cpu().numpy()
        labels = [sample.label for sample in selected_samples]
    
        # 使用 t-SNE 进行降维到 2D
        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
        tsne_results = tsne.fit_transform(embeds)
    
        # 映射 label 为数值（方便绘图）
        label_to_idx = {label: idx for idx, label in enumerate(visualize_classes)}
        label_idx = [label_to_idx[label] for label in labels]

        # 绘制 t-SNE 结果
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=label_idx, cmap='tab10', alpha=0.7)
    
        # 添加图例
        handles, _ = scatter.legend_elements()
        plt.legend(handles, visualize_classes, loc="upper right", title="Classes")

        plt.title(title)
        plt.xlabel("t-SNE component 1")
        plt.ylabel("t-SNE component 2")
        plt.grid(True)

        # 保存为 JPG 文件
        if save_path:
            plt.savefig(save_path, format='jpg')
            print(f"Figure saved as {save_path}")

    def run(self):
        results = {"avg":0}
        train_dataset = ImageNetDataset(
            root=os.path.join("/data/hyh/imagenet/data", "train"),
            index_file="./imagenet_class_indices.pkl"  # 指定索引文件路径
        )
        test_dataset = ImageNetDataset(os.path.join("/data/hyh/imagenet/data", "val"))

        if self.args.catergory_num == 100: # 测100类
            test_dataset = Subset(test_dataset, list(range(5000))) #取前5000个样本作为validate set
            self.all_class_names = IMAGENET_CLASSNAMES_100
        else:
            self.all_class_names = IMAGENET_CLASSNAMES

        print(f"self.args.catergory_num:{self.args.catergory_num}")
        print(f"len of self.all_class_names:{len(self.all_class_names)}")
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
            for class_id in tqdm(range(len(self.all_class_names)),desc="get samples for each class"):
                data_list = train_dataset.get_data_list_by_class(class_id=class_id)
                support_set.extend(data_list[0:self.args.M //len(self.all_class_names)])
                sample_pool.extend(data_list[50:150])
        else: # imbalanced
            num_classes = len(self.all_class_names)
            no_sample_classes = class_selection_rng.sample(range(num_classes), num_classes // 2)

            for i in range(len(self.all_class_names)):
                class_name = self.all_class_names[i]
                class_samples = train_dataset.get_data_list_by_class(class_name=class_name)
                # 样本池样本（从第10个到第110个，总共100个）
                sample_pool.extend(class_samples[100:200])
                if i not in no_sample_classes:
                    support_set.extend(class_samples[0:self.args.M*2//len(self.all_class_names)])           

        print(f"Support set size: {len(support_set)}, Sample pool size: {len(sample_pool)}")

        for idx in tqdm(range(len(support_set)), desc=f"Preprocess Supporting set..."):
            # 对sample进行预处理
            support_sample = self.preprocess_train(support_set[idx])
            self.retriever.demonstrations.append(support_sample)
           
            # 对 prototype 映射表 更新
            self.process_dict(support_sample)

        self._initialize_prototypes()

        # 预处理好了 self.retriever.demonstrations ，可以对其中的一些样本进行可视化了
        #self.visualize_tsne("Initial Memory Bank t-SNE", f"./{self.args.dataset_mode}_{self.args.update_strategy}-initial_memory_bank.jpg")
        
        print(f"Get the value of every sample in support set")
        sample_pool_rng.shuffle(sample_pool)  # 使用单独的 random 对象打乱 sample_pool

        shuffled_indices = list(range(len(test_dataset)))
        validate_rng.shuffle(shuffled_indices)

        # 使用数据流更新 support set
        #sample_pool=sample_pool[0:10]
        total_samples = len(sample_pool)  # 获取样本池的初始大小
        pbar = tqdm(total=total_samples, desc="Using sample pool to update the support set")
        while sample_pool:  # 当 sample_pool 不为空时继续循环
            sample = sample_pool.pop()
            #self.inference(sample)
            sample = self.preprocess_train(sample)
            self.retriever.update_online(sample)
            del sample
            pbar.update(1)  # 每处理一个样本，更新进度条

        self.predictions = []
        print(f"Successfully update the support set with sample pool, now support set size: {len(self.retriever.demonstrations)}")
        #self.visualize_tsne("Updated Memory Bank t-SNE", f"./{self.args.dataset_mode}_{self.args.update_strategy}-updated_memory_bank-alpha={self.args.alpha}.jpg")
        print("Inference using the latest supporting set...")

        self.test_sample_num = 0
        self.right_sample_num = 0

        for i in tqdm(range(0, len(test_dataset), self.args.batch_size), desc=f"Inference ImageNet..."):
            batch_indices = shuffled_indices[i:i + self.args.batch_size]
            batch_samples = [test_dataset[idx] for idx in batch_indices]
            self.inference_batch(batch_samples)
        
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
        self.features_data_train = pickle.load(open("/data/chy/feacture_cache/train_idx2embed_quality.pkl", 'rb'))
        self.features_data_val = pickle.load(open("/data/chy/feacture_cache/val_idx2embed.pkl",'rb'))

    def get_embedding(self, image):
        inputs = self.embedding_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.embedding_model.get_image_features(**inputs)
        return image_features

    def evaluate_batch_on_OFv2(self, batch_samples):
        batch_images = []
        batch_text = []
        batch_demonstrations = []
        for sample in batch_samples: # 遍历每个sample，找到分别对应的context
            ice_img,ice_text,demonstrations = self.retriever.get_final_query(sample)
            batch_images.append(ice_img)
            batch_text.append(ice_text)
            batch_demonstrations.append(demonstrations)

        batch_images = self._prepare_images(batch_images)
        ctx_input_ids, ctx_attention_mask = self._prepare_text(batch_text)

        _lang_x = torch.cat([ctx_input_ids], dim=1)
        _attention_mask = torch.cat(
            [
                ctx_attention_mask,
            ],
            dim=1,
        )
        _vision_x = batch_images

        _lang_x = torch.cat([ctx_input_ids], dim=1)
        _attention_mask = torch.cat(
            [
                ctx_attention_mask,
            ],
            dim=1,
        )
        _vision_x = batch_images

        with torch.no_grad():
            outputs = self.model.generate(
                vision_x=_vision_x,
                lang_x=_lang_x,
                attention_mask=_attention_mask,
                max_new_tokens=20,
                min_new_tokens=1,
                num_beams=1,
                output_scores=True,
                return_dict_in_generate=True
            )

        # Tokenize class names
        classnames_tokens = self.tokenizer(self.all_class_names)["input_ids"]

        # Get top-k classifications for the batch
        predicted_classnames_batch, predicted_logprobs_batch, overall_log_probs = get_topk_classifications_batch(
            outputs,
            classnames_tokens,
            self.topk
        )

        # Process predictions for each sample
        for idx, sample in enumerate(batch_samples):
            y_i = sample.label
            gt_label_index = self.all_class_names.index(y_i)
            gt_label_confidence = overall_log_probs[idx, gt_label_index].item()

            self.predictions.append(
                {
                    "id": sample.idx,
                    "gt_label": y_i,
                    "pred_label": predicted_classnames_batch[idx][0],
                    "gt_id": sample.class_id,
                    "pred_score": predicted_logprobs_batch[idx][0],
                    "gt_score": gt_label_confidence,
                    "prompt_text": batch_text[idx],
                    "prompt_label": [dm.class_id for dm in demonstrations]
                }
            )
            sample.pred_score = predicted_logprobs_batch[idx][0]
            sample.pseudo_label = predicted_classnames_batch[idx][0]
            sample.gt_score = gt_label_confidence
        
    def _prepare_images(self, batch: List[List[Image.Image]]) -> torch.Tensor:
        """
        Convert images to tensors, reshape them, and stack them.
        Args:
            batch: A list of lists of images.
        Returns:
            preprocessed images (tensors) or None
                shape (B, T_img, F, C, H, W)
                None if no images in batch
        """
        images_per_example = max(len(x) for x in batch)  # 一个批量有多少图片
        batch_images = None
        for iexample, example in enumerate(batch):
            for iimage, image in enumerate(example):
                preprocessed = self.image_processor(image)

                if batch_images is None:
                    batch_images = torch.zeros(
                        (len(batch), images_per_example, 1) + preprocessed.shape,
                        dtype=preprocessed.dtype,
                    )
                batch_images[iexample, iimage, 0] = preprocessed
        if batch_images is not None:
            batch_images = batch_images.to(self.device).half()
        return batch_images

    def _prepare_text(
            self,
            batch: List[List[str]],
            padding="longest",
            truncation=True,
            max_length=2000,
        ):
            """
            Tokenize the text and stack them.
            Args:
                batch: A list of lists of strings.
            Returns:
                input_ids (tensor)
                    shape (B, T_txt)
                attention_mask (tensor)
                    shape (B, T_txt)
            """
            self.tokenizer.padding_side = "left"
            encodings = self.tokenizer(
                batch,
                padding=padding,
                truncation=truncation,
                return_tensors="pt",
                max_length=max_length,
            )
            input_ids, attention_mask = encodings["input_ids"], encodings["attention_mask"]
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            return input_ids, attention_mask

    def inference_batch(self,batch_samples):
        batch_samples = [self.preprocess_val(sample) for sample in batch_samples]
        self.test_sample_num += len(batch_samples)
        if self.args.model == "open_flamingo_9b":
            self.evaluate_batch_on_OFv2(batch_samples)
        elif self.args.model == "idefics_v2":
            self.evaluate_batch_on_idev2(batch_samples)
        for sample in batch_samples:
            if sample.pseudo_label == sample.label:
                self.right_sample_num += 1

    def inference(self, sample):
        sample = self.preprocess_train(sample)
        self.test_sample_num += 1
        if self.args.model == "open_flamingo_9b":
            self.get_response_OFv2(sample)
        if self.args.model == "idefics_v2":
            self.get_response_ideficsv2(sample)
        if sample.pseudo_label == sample.label:
            self.right_sample_num += 1

    def evaluate_batch_on_idev2(self,batch_samples):
        # Prepare prompts and image sets for batch processing
        prompts = []
        images = []
        
        for sample in batch_samples:
            demonstrations = self.retriever.get_demonstrations_from_bank(sample)
            prompt = ""
            image_set = []
            
            # Append demonstration images and labels
            if demonstrations is not None:
                for dm in demonstrations:
                    image_set.append(dm.image)
                    prompt += f"<image>Category:{dm.label}."
            
            # Append the main sample image and category prompt
            image_set.append(sample.image)
            prompt += "<image>Category:"
            
            # Collect prompts and images
            prompts.append(prompt)
            images.append(image_set)

        BAD_WORDS_IDS = self.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
        EOS_WORDS_IDS = [self.tokenizer.eos_token_id]
        
        inputs = self.processor(images=images, text=prompts, padding=True, truncation=True, return_tensors="pt").to("cuda")
       
        with torch.no_grad():
            outputs = self.model.generate(**inputs, bad_words_ids=BAD_WORDS_IDS,min_new_tokens=1, max_new_tokens=20,num_beams=1,
                        length_penalty=1.0,
                        output_scores=True,
                        return_dict_in_generate=True)

        classnames_tokens = self.tokenizer(self.all_class_names)["input_ids"]
        # Get top-k classifications for the batch
        predicted_classnames_batch, predicted_logprobs_batch, overall_log_probs = get_topk_classifications_batch(
            outputs,
            classnames_tokens,
            self.topk,
        )
        # Process predictions for each sample
        for idx, sample in enumerate(batch_samples):
            y_i = sample.label
            gt_label_index = self.all_class_names.index(y_i)
            gt_label_confidence = overall_log_probs[idx, gt_label_index].item()

            self.predictions.append(
                {
                    "id": sample.idx,
                    "gt_label": y_i,
                    "pred_label": predicted_classnames_batch[idx][0],
                    "gt_id": sample.class_id,
                    "pred_score": predicted_logprobs_batch[idx][0],
                    "gt_score": gt_label_confidence,
                    "prompt_label": [dm.label for dm in demonstrations]
                }
            )
            sample.pred_score = predicted_logprobs_batch[idx][0]
            sample.pseudo_label = predicted_classnames_batch[idx][0]
            sample.gt_score = gt_label_confidence

    def preprocess_train(self, sample):
        idx = sample["id"]  
        image = sample["image"]
        label = sample["class_name"]
        class_id = sample["class_id"]
        embed, quality = self.features_data_train[idx]
        sample = Sample(idx, image, label, embed,quality,class_id, None,None,None,None)
        return sample
    
    def preprocess_val(self, sample):
        idx = sample["id"]
        image = sample["image"]
        label = sample["class_name"]
        class_id = sample["class_id"]
        embed= self.features_data_val[idx]
        sample = Sample(idx, image, label, embed,None,class_id, None,None,None,None)
        return sample
    
    def run(self):
        results = {"avg": 0}
        train_dataset = ImageNetDataset(
            root=os.path.join("/data/hyh/imagenet/data", "train"),
            index_file="./imagenet_class_indices.pkl"  # 指定索引文件路径
        )
        test_dataset = ImageNetDataset(os.path.join("/data/hyh/imagenet/data", "val"))
        # 设置全局随机种子
        random.seed(self.args.seed)
        if self.args.catergory_num == 100: # 测100类
            test_dataset = Subset(test_dataset, list(range(5000))) #取前5000个样本作为validate set
            self.all_class_names = IMAGENET_CLASSNAMES_100
        else:
            self.all_class_names = IMAGENET_CLASSNAMES
        # 创建不同的随机数生成器
        validate_rng = random.Random(self.args.seed)
        class_selection_rng = random.Random(self.args.seed + 3)
        # 从train_dataset中取support set
        support_set = []
        #sample_pool=[]
        print("get supportng set ...")
        if self.args.dataset_mode == "balanced":
            for class_id in tqdm(range(len(self.all_class_names)),desc="get samples for each class"):
                data_list = train_dataset.get_data_list_by_class(class_id=class_id)
                support_set.extend(data_list[0:10])
                #support_set.extend(data_list[50:150])
              
        # 输出支持集 大小以确认正确性
        print(f"Support set size: {len(support_set)}")

        for idx in tqdm(range(len(support_set)), desc=f"Preprocess Supporting set..."):
            # 对sample进行预处理
            support_sample = self.preprocess_train(support_set[idx])
            self.retriever.demonstrations.append(support_sample)

        shuffled_indices = list(range(len(test_dataset)))
        validate_rng.shuffle(shuffled_indices)

        self.test_sample_num = 0
        self.right_sample_num = 0
        
        batch_size = self.args.batch_size  

        for i in tqdm(range(0, len(test_dataset), batch_size), desc=f"Inference ImageNet..."):
            batch_indices = shuffled_indices[i:i + batch_size]
            batch_samples = [test_dataset[idx] for idx in batch_indices]
            self.inference_batch(batch_samples)

        acc = self.right_sample_num / self.test_sample_num
        results["avg"] += acc
        return results,self.predictions
