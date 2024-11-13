import dataclasses
import retriever
from retriever import DynamicReteiever
from tqdm import tqdm
import torch
from imagenet_dataset import ImageNetDataset
import os
from PIL import Image
from classification_utils import IMAGENET_CLASSNAMES_100,IMAGENET_CLASSNAMES
from torch.utils.data import Subset
from typing import Optional
from utils import *
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
    image: Optional[Image.Image]
    label: str
    embed: torch.Tensor
    feature_256_1024: torch.Tensor
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

    def __init__(self, args, tokenizer, model, image_processor,device='cuda'):
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.device = device
        self.test_sample_num = 0
        self.right_sample_num = 0
        self.all_class_names = IMAGENET_CLASSNAMES_100
        self.retriever = DynamicReteiever(args)
        self.predictions = []
        self.topk = 1
        self.no_kv_caching = False
        self.features_data_train = pickle.load(open("/path/to/train_idx2embed_quality.pkl", 'rb'))
        self.features_data_val = pickle.load(open("/path/to/val_idx2embed.pkl",'rb'))
        self.features_data_train_256_1024 = pickle.load(open("/path/to/train_features_256x1024.pkl", 'rb'))
        self.features_data_val_256_1024 = pickle.load(open("/path/to/val_features_256x1024.pkl", 'rb'))

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
        lang_x = self.tokenizer(ice_text, return_tensors="pt")
        lang_x = {k: v.to(self.device) for k, v in lang_x.items()}
        return lang_x

    def get_response_OFv2(self, sample):
        ice_img,ice_text,demonstrations = self.retriever.get_final_query(sample)

        # 将列表中的 tensor 堆叠成 (5, 256, 1024)
        ice_img_tensor = torch.stack(ice_img)  
        
        ice_img_tensor = ice_img_tensor.unsqueeze(0) # (1, 5,256, 1024)
        
        # 添加一个维度到第2维
        vision_features = ice_img_tensor.unsqueeze(2).to(self.device)  # 变为 (1, 5, 1, 64, 1024)

        lang_x = self.prepare_text(ice_text)
        with torch.no_grad():
            outputs = self.model.generate(
                vision_x=None,
                lang_x=lang_x["input_ids"],
                attention_mask=lang_x["attention_mask"],
                vision_features=vision_features,
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
    
    def inference(self, sample):
        sample = self.preprocess_train(sample)
        self.test_sample_num += 1
        if self.args.model == "open_flamingo_9b" or self.args.model == "open_flamingo_3b":
            self.get_response_OFv2(sample)
        if sample.pseudo_label == sample.label:
            self.right_sample_num += 1
        self.retriever.update_online(sample)
        
    def evaluate_batch_on_OFv2(self, batch_samples):
        batch_images = []
        batch_text = []
        batch_demonstrations = []
        for sample in batch_samples: # 遍历每个sample，找到分别对应的context
            ice_img,ice_text,demonstrations = self.retriever.get_final_query(sample)
            # 将列表中的 tensor 堆叠成 (5, 256, 1024)
            ice_img_tensor = torch.stack(ice_img)  # 形状变为 (5, 256, 1024)
            batch_images.append(ice_img_tensor)
            batch_text.append(ice_text)
            batch_demonstrations.append(demonstrations)
        #batch_images = self._prepare_images(batch_images)
        # 将所有的 (5, 256, 1024) tensor 堆叠成 (batch_size, 5, 256, 1024)
        batch_images = torch.stack(batch_images)  # 形状变为 (batch_size, 5, 64, 1024)
        # 添加一个维度到第一维
        batch_images = batch_images.unsqueeze(2).to(self.device)  # 变为 (batch_size, 5, 1, 256, 1024)
        
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
                vision_x=None,  # 因为我们使用 vision_features，所以 vision_x 可以为 None
                lang_x=_lang_x,
                attention_mask=_attention_mask,
                vision_features=batch_images,
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
            outputs = self.model.generate(**inputs, bad_words_ids=BAD_WORDS_IDS,min_new_tokens=1, max_new_tokens=7,num_beams=1,
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

    def inference_batch(self,batch_samples):
        batch_samples = [self.preprocess_val(sample) for sample in batch_samples]
        self.test_sample_num += len(batch_samples)
        self.evaluate_batch_on_OFv2(batch_samples)
        for sample in batch_samples:
            if sample.pseudo_label == sample.label:
                self.right_sample_num += 1

    def preprocess_train(self, sample):
        idx = sample["id"]  
        image = sample["image"]
        label = sample["class_name"]
        class_id = sample["class_id"]
        embed, quality = self.features_data_train[idx]
        feature_256_1024 = self.features_data_train_256_1024[idx]
        sample = Sample(idx, None, label, embed,feature_256_1024,quality,class_id, None,None,None,None)
        return sample
    
    def preprocess_val(self, sample):
        idx = sample["id"]
        image = sample["image"]
        label = sample["class_name"]
        class_id = sample["class_id"]
        embed= self.features_data_val[idx]
        feature_256_1024 = self.features_data_val_256_1024[idx]
        sample = Sample(idx, None, label, embed,feature_256_1024,None,class_id, None,None,None,None)
        return sample
    
    def process_dict(self,sample):
        if sample.label not in self.retriever.label2sample:
            self.retriever.label2sample[sample.label] = [sample]
        else:
            self.retriever.label2sample[sample.label].append(sample)

    def store_bank(self, output_pkl_path):
        sample_list = {}

        for i in tqdm(range(0, len(self.retriever.demonstrations)), desc="Store bank..."):
            sample = self.retriever.demonstrations[i]
            label = sample.label

            # 如果类别不存在于字典中，则初始化该类别的列表
            if label not in sample_list:
                sample_list[label] = [sample.feature_256_1024.cpu()]
            else:
                sample_list[label].append(sample.feature_256_1024.cpu())

        # 将合并后的字典保存为 .pkl 文件
        with open(output_pkl_path, 'wb') as f:
            pickle.dump(sample_list, f)
            print(f"Samples saved to {output_pkl_path}")

    def _initialize_prototypes(self):
        default_prototype_shape = (256, 1024)
        default_zero_prototype = torch.zeros(default_prototype_shape)
        
        for label, samples in self.retriever.label2sample.items():
            if samples:
                embeddings = torch.stack([s.feature_256_1024 for s in samples])
                prototype = torch.mean(embeddings, dim=0)
                self.retriever.label_to_prototype[label] = prototype
            else:
                # 如果没有样本，使用默认的零张量
                self.retriever.label_to_prototype[label] = default_zero_prototype.clone()

    def run(self):
        results = {"avg":0}
        train_dataset = ImageNetDataset(
            root=os.path.join("/path/to/imagenet/data", "train"),
            index_file="./imagenet_class_indices.pkl"  # 指定索引文件路径
        )
        test_dataset = ImageNetDataset(os.path.join("/path/to/imagenet/data", "val"))

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
        for class_id in tqdm(range(len(self.all_class_names)),desc="get samples for each class"):
            data_list = train_dataset.get_data_list_by_class(class_id=class_id)
            support_set.extend(data_list[0:self.args.M //len(self.all_class_names)])
            sample_pool.extend(data_list[50:50+self.args.stream//len(self.all_class_names)])
            
        print(f"Support set size: {len(support_set)}, Sample pool size: {len(sample_pool)}")

        for idx in tqdm(range(len(support_set)), desc=f"Preprocess Supporting set..."):
            # 对sample进行预处理
            support_sample = self.preprocess_train(support_set[idx])
            self.retriever.demonstrations.append(support_sample)
           
            # 对 prototype 映射表 更新
            self.process_dict(support_sample)

        self._initialize_prototypes()
        train_dataset = None
        print(f"Get the value of every sample in support set")
        sample_pool_rng.shuffle(sample_pool)  # 使用单独的 random 对象打乱 sample_pool

        shuffled_indices = list(range(len(test_dataset)))
        validate_rng.shuffle(shuffled_indices)

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
    def __init__(self, args, tokenizer, model, image_processor,device):
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.device = device
        self.test_sample_num = 0
        self.right_sample_num = 0
        self.all_class_names = IMAGENET_CLASSNAMES_100
        self.retriever = DynamicReteiever(args)
        self.predictions = []
        self.no_kv_caching = False
        self.topk = 1       
        self.features_data_train = pickle.load(open("/path/to/train_idx2embed_quality.pkl", 'rb'))
        self.features_data_val = pickle.load(open("/path/to/val_idx2embed.pkl",'rb'))
        self.features_data_train_256_1024 = pickle.load(open("/path/to/train_features_256x1024.pkl", 'rb'))
        self.features_data_val_256_1024 = pickle.load(open("/path/to/val_features_256x1024.pkl", 'rb'))


    def get_embedding(self, image):
        inputs = self.embedding_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.embedding_model.get_image_features(**inputs)
        return image_features

    def evaluate_batch_on_OFv2(self, batch_samples):
        batch_images = []
        batch_text = []
        for sample in batch_samples: # 遍历每个sample，找到分别对应的context
            ice_img,ice_text,demonstrations = self.retriever.get_final_query(sample)
            ice_img_tensor = torch.stack(ice_img)  # 形状变为 (5, 64, 1024)
            batch_images.append(ice_img_tensor)
            batch_text.append(ice_text)

        #batch_images = self._prepare_images(batch_images)
        # 将所有的 (5, 64, 1024) tensor 堆叠成 (batch_size, 5, 64, 1024)
        batch_images = torch.stack(batch_images)  # 形状变为 (batch_size, 5, 64, 1024)
        batch_images = batch_images.unsqueeze(2).to(self.device)  # 变为 (batch_size, 5, 1, 256, 1024)
        ctx_input_ids, ctx_attention_mask = self._prepare_text(batch_text)

        _lang_x = torch.cat([ctx_input_ids], dim=1)
        _attention_mask = torch.cat(
            [
                ctx_attention_mask,
            ],
            dim=1,
        )
    
        _vision_x = None

        with torch.no_grad():
            outputs = self.model.generate(
                vision_x=_vision_x,
                lang_x=_lang_x,
                attention_mask=_attention_mask,
                vision_features = batch_images,
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
        self.evaluate_batch_on_OFv2(batch_samples)
        for sample in batch_samples:
            if sample.pseudo_label == sample.label:
                self.right_sample_num += 1

    def preprocess_train(self, sample):
        idx = sample["id"]  
        image = sample["image"]
        label = sample["class_name"]
        class_id = sample["class_id"]
        embed, quality = self.features_data_train[idx]
        feature_256_1024 = self.features_data_train_256_1024[idx]
        sample = Sample(idx, None, label, embed,feature_256_1024,quality,class_id, None,None,None,None)
        return sample
    
    def preprocess_val(self, sample):
        idx = sample["id"]
        image = sample["image"]
        label = sample["class_name"]
        class_id = sample["class_id"]
        embed= self.features_data_val[idx]
        feature_256_1024 = self.features_data_val_256_1024[idx]
        sample = Sample(idx, None, label, embed,feature_256_1024,None,class_id, None,None,None,None)
        return sample
    
    def run(self):
        results = {"avg": 0}
        train_dataset = ImageNetDataset(
            root=os.path.join("/path/to/imagenet/data", "train"),
            index_file="./imagenet_class_indices.pkl"  # 指定索引文件路径
        )
        test_dataset = ImageNetDataset(os.path.join("/path/to/imagenet/data", "val"))
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
                support_set.extend(data_list[0:self.args.M//len(self.all_class_names)])
                if self.args.bank == "total":
                    support_set.extend(data_list[50:50+self.args.stream//len(self.all_class_names)])
        
        train_dataset = None
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
        
        # for idx in tqdm(range(0, len(test_dataset)), desc=f"Inference ImageNet..."):
        #     self.inference(test_dataset[idx])

        batch_size = self.args.batch_size  

        for i in tqdm(range(0, len(test_dataset), batch_size), desc=f"Inference ImageNet..."):
            batch_indices = shuffled_indices[i:i + batch_size]
            batch_samples = [test_dataset[idx] for idx in batch_indices]
            self.inference_batch(batch_samples)

        acc = self.right_sample_num / self.test_sample_num
        results["avg"] += acc
        return results,self.predictions
