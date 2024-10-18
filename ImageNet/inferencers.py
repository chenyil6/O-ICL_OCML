import dataclasses
from retriever import DynamicReteiever
from tqdm import tqdm
import torch
from imagenet_dataset import ImageNetDataset
import os
from PIL import Image
from classification_utils import IMAGENET_CLASSNAMES_100,IMAGENET_CLASSNAMES
from torch.utils.data import Subset
from typing import Optional
from utils import get_topk_classifications,prepare_eval_samples
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import json
import pickle
import logging
logger = logging.getLogger(__name__)

@dataclasses.dataclass
class Sample:
    idx: int
    image: Image
    label:str
    embed: torch.TensorType
    quality:Optional[float]
    class_id:int
    pseudo_label: Optional[str]
    pred_score: Optional[float]
    gt_score:Optional[float]

class Online_ICL_Old:
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
        self.batch_size = 32
        self.features_data_train = {}
        self.features_data__train_file = "./train_idx2embed_quality.pkl"
        self.features_data_val = {}
        self.features_data__val_file = "./val_idx2embed.pkl"

        if os.path.exists(self.features_data__train_file):
            with open(self.features_data__train_file, 'rb') as f:
                self.features_data_train = pickle.load(f)

        if os.path.exists(self.features_data__val_file):
            with open(self.features_data__val_file, 'rb') as f:
                self.features_data_val = pickle.load(f)


    def get_image_embedding(self, image):
        inputs = self.embedding_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.embedding_model.get_image_features(**inputs)
        return image_features
    
    def get_text_embedding(self, label):
        inputs = self.embedding_tokenizer(text=label, padding=True,return_tensors="pt")
        with torch.no_grad():
            text_features = self.embedding_model.get_text_features(**inputs)
        return text_features

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

        predicted_classnames,predicted_logprobs,overall_log_probs = get_topk_classifications(outputs,classnames_tokens,self.topk)
        # compute accuracy
        y_i = sample.label

        # Find the index of the ground truth label
        gt_label_index = self.all_class_names.index(y_i)

        # Get the confidence score of the ground truth label
        gt_label_confidence = overall_log_probs[gt_label_index].item()
        self.predictions.append(
            {
                "id": sample.idx,
                "gt_label": y_i,
                "pred_label": predicted_classnames[0],
                "gt_id": sample.class_id,
                "pred_score": predicted_logprobs[0],
                "gt_score": gt_label_confidence,  
                "prompt_text":ice_text,
                "prompt_label":[dm.class_id for dm in demonstrations]
            }
        )
        sample.pred_score = predicted_logprobs[0]
        sample.pseudo_label = predicted_classnames[0]
        sample.gt_score = gt_label_confidence 

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
        sample = self.preprocess_val(sample)
        self.test_sample_num += 1
        if self.args.model == "open_flamingo":
            self.get_response_OFv2(sample)
        if self.args.model == "idefics":
            self.get_response_idefics(sample)
        if sample.pseudo_label == sample.label:
            self.right_sample_num += 1

    def preprocess_train(self, sample):
        idx = sample["id"]
        image = sample["image"]
        label = sample["class_name"]
        class_id = sample["class_id"]
        embed,quality = self.features_data_train[idx]
        sample = Sample(idx, image, label, embed,quality,class_id, None,None,None)
        return sample
    
    def preprocess_val(self, sample):
        idx = sample["id"]
        image = sample["image"]
        label = sample["class_name"]
        class_id = sample["class_id"]
        embed = self.features_data_val[idx]
        sample = Sample(idx, image, label, embed,None,class_id, None,None,None)
        return sample
    
    def process_dict(self,sample):
        if sample.label not in self.retriever.label2sample:
            self.retriever.label2sample[sample.label] = [sample]
        else:
            self.retriever.label2sample[sample.label].append(sample)

    def _initialize_prototypes(self,):
        for label, samples in self.retriever.label2sample.items():
            if samples:
                embeddings = torch.stack([s.embed for s in samples])
                prototype = torch.mean(embeddings, dim=0)
                self.retriever.label_to_prototype[label] = prototype
            else:
                self.retriever.label_to_prototype[label] = torch.zeros_like(next(iter(self.label_to_prototype.values())))

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
        train_dataset = ImageNetDataset(os.path.join("/data/hyh/imagenet/data", "train"))
        test_dataset = ImageNetDataset(os.path.join("/data/hyh/imagenet/data", "val"))
        validate_set = Subset(test_dataset, list(range(5000))) #取前5000个样本作为validate set
        # 设置全局随机种子
        random.seed(self.args.seed)
    
        validate_rng = random.Random(self.args.seed)  # 用于打乱 validate_set 的随机生成器
        sample_pool_rng = random.Random(self.args.seed + 1)  # 用于打乱 sample_pool 的随机生成器
        class_selection_rng = random.Random(self.args.seed + 3)  # 用于选择 no_sample_classes 的随机生成器
    
        support_set = []
        sample_pool = []

        print("get memory bank and sample pool ...")
        if self.args.dataset_mode == "balanced":
            for class_id in tqdm(range(len(self.all_class_names)),desc="get samples for each class"):
                data_list = train_dataset.get_data_list_by_class(class_id=class_id)
                support_set.extend(data_list[0:10])
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
                    support_set.extend(class_samples[0:self.args.M*2//100])           

        print(f"Support set size: {len(support_set)}, Sample pool size: {len(sample_pool)}")

        for idx in tqdm(range(len(support_set)), desc=f"Preprocess Supporting set..."):
            # 对sample进行预处理
            support_sample = self.preprocess(support_set[idx])
            self.retriever.demonstrations.append(support_sample)
            self.process_dict(support_sample)

        self._initialize_prototypes()

        # 预处理好了 self.retriever.demonstrations ，可以对其中的一些样本进行可视化了
        self.visualize_tsne("Initial Memory Bank t-SNE", f"./{self.args.dataset_mode}_{self.args.update_strategy}-initial_memory_bank.jpg")
        
        print(f"Get the value of every sample in support set")
        sample_pool_rng.shuffle(sample_pool)  # 使用单独的 random 对象打乱 sample_pool

        for idx in tqdm(range(len(sample_pool)), desc=f"Preprocess Sample Pool ..."):
            # 对sample进行预处理
            sample_pool_sample = self.preprocess(sample_pool[idx])
            self.retriever.pool.append(sample_pool_sample)

        # 打乱 validate_set 的索引
        shuffled_validate_indices = list(range(len(validate_set)))
        validate_rng.shuffle(shuffled_validate_indices)
        # 创建一个新的 Subset 作为打乱后的 validate_dataset
        validate_dataset = Subset(test_dataset, shuffled_validate_indices)

        self.test_sample_num = 0
        self.right_sample_num = 0

        for idx in tqdm(range(len(sample_pool)), desc=f"Updating the support set..."):
            self.retriever.update()

        print(f"Successfully update the support set with sample pool")
        self.visualize_tsne("Updated Memory Bank t-SNE", f"./{self.args.dataset_mode}_{self.args.update_strategy}-updated_memory_bank.jpg")
        print("Inference using the latest supporting set...")
        
        self.test_sample_num = 0
        self.right_sample_num = 0
        for idx in tqdm(range(len(validate_dataset)), desc=f"Inference Support Set..."):
            self.inference(validate_dataset[idx])
        acc = self.right_sample_num / self.test_sample_num
        results["avg"] += acc

        return results,self.predictions
    

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
        self.features_data_train = {}
        self.features_data__train_file = "./train_idx2embed_quality.pkl"
        self.features_data_val = {}
        self.features_data__val_file = "./val_idx2embed_quality.pkl"

        if os.path.exists(self.features_data__train_file):
            with open(self.features_data__train_file, 'rb') as f:
                self.features_data_train = pickle.load(f)

        if os.path.exists(self.features_data__val_file):
            with open(self.features_data__val_file, 'rb') as f:
                self.features_data_val = pickle.load(f)

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

        predicted_classnames,predicted_logprobs,overall_log_probs = get_topk_classifications(outputs,classnames_tokens,self.topk)
        # compute accuracy
        y_i = sample.label

        # Find the index of the ground truth label
        gt_label_index = self.all_class_names.index(y_i)

        # Get the confidence score of the ground truth label
        gt_label_confidence = overall_log_probs[gt_label_index].item()

        self.predictions.append(
            {
                "id": sample.idx,
                "gt_label": y_i,
                "pred_label": predicted_classnames[0],
                "gt_id": sample.class_id,
                "pred_score": predicted_logprobs[0],
                "gt_score": gt_label_confidence,  
                "prompt_text":ice_text,
                "prompt_label":[dm.class_id for dm in demonstrations]
            }
        )
        sample.pred_score = predicted_logprobs[0]
        sample.pseudo_label = predicted_classnames[0]
        sample.gt_score = gt_label_confidence 

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
        #sample = self.preprocess(sample)
        self.test_sample_num += 1
        if self.args.model == "open_flamingo":
            self.get_response_OFv2(sample)
        if self.args.model == "idefics":
            self.get_response_idefics(sample)
        if sample.pseudo_label == sample.label:
            self.right_sample_num += 1
        self.retriever.update_online(sample)

    def preprocess_train(self, sample):
        idx = sample["id"]
        image = sample["image"]
        label = sample["class_name"]
        class_id = sample["class_id"]
        embed, quality = self.features_data_train[idx]
        sample = Sample(idx, image, label, embed,quality,class_id, None,None,None)
        return sample
    
    def preprocess_val(self, sample):
        idx = sample["id"]
        image = sample["image"]
        label = sample["class_name"]
        class_id = sample["class_id"]
        # embed = self.get_embedding(image).squeeze().cpu()
        # label_embed = self.get_text_embedding(label).squeeze().cpu()
        # quality = torch.cosine_similarity(embed.unsqueeze(0), label_embed.unsqueeze(0), dim=1).item()
        embed = self.features_data_val[idx]
        sample = Sample(idx, image, label, embed,None,class_id, None,None,None)
        return sample
    
    def process_dict(self,sample):
        if sample.label not in self.retriever.label2sample:
            self.retriever.label2sample[sample.label] = [sample]
        else:
            self.retriever.label2sample[sample.label].append(sample)

    def _initialize_prototypes(self,):
        for label, samples in self.retriever.label2sample.items():
            if samples:
                embeddings = torch.stack([s.embed for s in samples])
                prototype = torch.mean(embeddings, dim=0)
                self.retriever.label_to_prototype[label] = prototype
            else:
                self.retriever.label_to_prototype[label] = torch.zeros_like(next(iter(self.label_to_prototype.values())))

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
                support_set.extend(data_list[0:10])
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
        self.visualize_tsne("Initial Memory Bank t-SNE", f"./{self.args.dataset_mode}_{self.args.update_strategy}-initial_memory_bank.jpg")
        
        print(f"Get the value of every sample in support set")
        sample_pool_rng.shuffle(sample_pool)  # 使用单独的 random 对象打乱 sample_pool

        #for idx in tqdm(range(len(sample_pool)), desc=f"Preprocess Sample Pool ..."):
            # 对sample进行预处理
            #sample_pool_sample = self.preprocess(sample_pool[idx])
            #self.retriever.pool.append(sample_pool_sample)

        # 打乱 validate_set 的索引
        shuffled_validate_indices = list(range(len(test_dataset)))
        validate_rng.shuffle(shuffled_validate_indices)
        # 创建一个新的 Subset 作为打乱后的 validate_dataset
        validate_dataset = Subset(test_dataset, shuffled_validate_indices)

        # 需要对数据流的数据做推理
        for idx in tqdm(range(len(sample_pool)), desc=f"inferenc sample pool meanwhile updating the support set..."):
            sample = sample_pool[idx]
            sample = self.preprocess_train(sample)
            self.inference(sample)

        self.predictions = []
        print(f"Successfully update the support set with sample pool, now support set size: {len(self.retriever.demonstrations)}")
        self.visualize_tsne("Updated Memory Bank t-SNE", f"./{self.args.dataset_mode}_{self.args.update_strategy}-updated_memory_bank-alpha={self.args.alpha}-beta={self.args.beta}-delta={self.args.delta}.jpg")
        print("Inference using the latest supporting set...")

        file_path = f'./{self.args.update_strategy}-gradientUpdate-alpha={self.args.alpha}-beta={self.args.beta}-delta={self.args.delta}.json'  # 确保 self.args.update_strategy 有效
        with open(file_path, 'w') as json_file:  # 确保 file_path 和 json_file 在这里定义
            json.dump(self.retriever.support_gradient_list, json_file, indent=4)

        self.test_sample_num = 0
        self.right_sample_num = 0
        for idx in tqdm(range(len(validate_dataset)), desc=f"Inference Support Set..."):
            sample = self.preprocess_val(validate_dataset[idx])
            self.inference(sample)
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
        self.all_class_names = IMAGENET_CLASSNAMES
        self.retriever = DynamicReteiever(args)
        self.predictions = []
        self.topk = 1       
        self.features_data_train = {}
        self.features_data__train_file = "./train_idx2embed_quality.pkl"
        self.features_data_val = {}
        self.features_data__val_file = "./val_idx2embed.pkl"

        if os.path.exists(self.features_data__train_file):
            with open(self.features_data__train_file, 'rb') as f:
                self.features_data_train = pickle.load(f)

        if os.path.exists(self.features_data__val_file):
            with open(self.features_data__val_file, 'rb') as f:
                self.features_data_val = pickle.load(f)

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
        # Log the device information for vision_x
        logger.info(f"vision_x is on device: {vision_x.device}")
        return vision_x

    def prepare_text(self,ice_text):
        self.tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
        lang_x = self.tokenizer(ice_text, return_tensors="pt", )
        lang_x = {k: v.to(self.device) for k, v in lang_x.items()}
        logger.info(f"lang_x is on device: {next(iter(lang_x.values())).device}")
        return lang_x

    def get_response_OFv2(self, sample):
        ice_img,ice_text,demonstrations = self.retriever.get_final_query(sample)
        vision_x = self.prepare_image(ice_img)
        lang_x = self.prepare_text(ice_text)
        # Check and log if the model is on the correct device
        model_device = next(self.model.parameters()).device
        logger.info(f"Model is on device: {model_device}")
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

        predicted_classnames,predicted_logprobs,overall_log_probs = get_topk_classifications(outputs,classnames_tokens,self.topk)
        # compute accuracy
        y_i = sample.label

        # Find the index of the ground truth label
        gt_label_index = self.all_class_names.index(y_i)

        # Get the confidence score of the ground truth label
        gt_label_confidence = overall_log_probs[gt_label_index].item()

        self.predictions.append(
            {
                "id": sample.idx,
                "gt_label": y_i,
                "pred_label": predicted_classnames[0],
                "gt_id": sample.class_id,
                "pred_score": predicted_logprobs[0],
                "gt_score": gt_label_confidence,  
                "prompt_text":ice_text,
                "prompt_label":[dm.class_id for dm in demonstrations]
            }
        )
        sample.pred_score = predicted_logprobs[0]
        sample.pseudo_label = predicted_classnames[0]
        sample.gt_score = gt_label_confidence 

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
        sample = self.preprocess_val(sample)
        self.test_sample_num += 1
        if self.args.model == "open_flamingo":
            self.get_response_OFv2(sample)
        if self.args.model == "idefics":
            self.get_response_idefics(sample)
        if sample.pseudo_label == sample.label:
            self.right_sample_num += 1

    def preprocess_train(self, sample):
        idx = sample["id"]
        image = sample["image"]
        label = sample["class_name"]
        class_id = sample["class_id"]
        embed, quality = self.features_data_train[idx]
        sample = Sample(idx, image, label, embed,quality,class_id, None,None,None)
        return sample
    
    def preprocess_val(self, sample):
        idx = sample["id"]
        image = sample["image"]
        label = sample["class_name"]
        class_id = sample["class_id"]
        embed= self.features_data_val[idx]
        sample = Sample(idx, image, label, embed,None,class_id, None,None,None)
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
    
        # 创建不同的随机数生成器
        validate_rng = random.Random(self.args.seed)
        class_selection_rng = random.Random(self.args.seed + 3)
        # 从train_dataset中取support set
        support_set = []
        sample_pool=[]
        print("get supportng set ...")
        if self.args.dataset_mode == "balanced":
            for class_id in tqdm(range(len(self.all_class_names)),desc="get samples for each class"):
                data_list = train_dataset.get_data_list_by_class(class_id=class_id)
                support_set.extend(data_list[0:10])
                sample_pool.extend(data_list[50:150])
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
                    
        # 输出支持集 大小以确认正确性
        print(f"Support set size: {len(support_set)}")

        for idx in tqdm(range(len(support_set)), desc=f"Preprocess Supporting set..."):
            # 对sample进行预处理
            support_sample = self.preprocess_train(support_set[idx])
            self.retriever.demonstrations.append(support_sample)

        # 打乱 validate_set 的索引
        shuffled_indices = list(range(len(test_dataset)))
        validate_rng.shuffle(shuffled_indices)

        self.test_sample_num = 0
        self.right_sample_num = 0
        for idx in tqdm(shuffled_indices, desc="Inference ImageNet..."):
            validate_sample = test_dataset[idx]
            self.inference(validate_sample)

        acc = self.right_sample_num / self.test_sample_num
        results["avg"] += acc
        return results,self.predictions
