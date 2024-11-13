import torch
import random
from utils import get_imagenet_prompt
import os
import json
import numpy as np
import time
from collections import defaultdict, deque
import logging
import torch.nn.functional as F
import math
logger = logging.getLogger(__name__)

class DynamicReteiever:
    def __init__(self, args):
        self.args = args
        self.demonstrations = []
        self.label2sample = dict()
        self.label_to_prototype = {}
        self.error_history = defaultdict(lambda: deque(maxlen=10))
        self.support_gradient_list = []
        self.class_gradients = defaultdict(list)
        with open('/path/to/test_idx2_ice_idx_1w.json', 'r') as file:
            self.test_idx2_ice_idx = json.load(file)

    def get_final_query(self, sample):
        demonstrations = self.get_demonstrations_from_bank(sample)
        ice_text = ""
        ice_img=[]
        if demonstrations is not None:
            for dm in demonstrations:
                ice_img.append(dm.feature_256_1024)
                ice_text += get_imagenet_prompt(dm.label)
                    
        ice_img.append(sample.feature_256_1024)
        ice_text += get_imagenet_prompt()
        return ice_img,ice_text,demonstrations

    def get_demonstrations_from_bank(self, sample):
        if self.args.dnum == 0:
            return []
        if self.args.select_strategy == "random":
            indices = self.get_random(sample)
        elif self.args.select_strategy == "cosine":
            indices = self.get_topk_cosine(sample)
        else:
            print("select_strategy is not effective.")
            return
        return [self.demonstrations[i] for i in indices]

    def get_random(self, sample):
        indices = random.sample(range(len(self.demonstrations)), self.args.dnum)
        return indices

    def get_topk_cosine(self, sample):
        if self.args.bank == "total" :
            # 直接用json文件中的索引
            indices = self.test_idx2_ice_idx[str(sample.idx)][0:self.args.dnum]
        else:
            demonstration_embeds = torch.stack([sample.feature_256_1024 for sample in self.demonstrations], dim=0)
            device_set = "cuda:" + str(self.args.device)
            device = torch.device(device_set)
            demonstration_embeds = demonstration_embeds.to(device)
            sample_embed = sample.feature_256_1024.to(device)
            #sample_embed = sample.feature_256_1024
            scores = torch.cosine_similarity(demonstration_embeds, sample_embed.unsqueeze(0), dim=-1)
            mean_scores = scores.mean(dim=1)
            values, indices = torch.topk(mean_scores, self.args.dnum, largest=True)
            indices = indices.cpu().tolist() 
            self.test_idx2_ice_idx[sample.idx] = indices
        return indices
    
    def update_online(self,query_sample):
        device_set = "cuda:" + str(self.args.device)
        device = torch.device(device_set)
        label = query_sample.label
        
        # 获取当前类别的样本列表
        sample_list = self.label2sample[label]

        if self.args.target_select == "prototype":
            # 获取当前类别的原型向量
            current_prototype = self.label_to_prototype[label] #(256,1024)
            current_prototype = current_prototype.to(device)
        
            embeddings = torch.stack([s.feature_256_1024 for s in sample_list])
            embeddings = embeddings.to(device)

            similarities = F.cosine_similarity(embeddings, current_prototype.unsqueeze(0), dim=-1)
            mean_similarities = similarities.mean(dim=-1)  # (N,)
            least_similar_index = torch.argmin(mean_similarities).cpu().item()

            target_sample = sample_list[least_similar_index]
        elif self.args.target_select == "random":
            # 随机选择当前类别的一个样本作为目标样本
            target_sample = random.choice(self.label2sample[query_sample.label])

        elif self.args.target_select == "most_similarity":
            query_embedding = query_sample.feature_256_1024.to(device)  # (256, 1024)

            # 获取当前类别所有样本的特征
            embeddings = torch.stack([s.feature_256_1024 for s in sample_list])  # (N, 256, 1024)
            embeddings = embeddings.to(device)

            # 计算与 query_sample 的余弦相似度
            similarities = F.cosine_similarity(embeddings, query_embedding.unsqueeze(0), dim=-1)  # (N, 256)
            mean_similarities = similarities.mean(dim=-1)  # 每个样本与 query_sample 的平均相似度 (N,)
            
            # 找到最相似的样本
            most_similar_index = torch.argmax(mean_similarities).cpu().item()
            target_sample = sample_list[most_similar_index]
        elif self.args.target_select == "least_similarity":
            query_embedding = query_sample.feature_256_1024.to(device)  # (256, 1024)

            # 获取当前类别所有样本的特征
            embeddings = torch.stack([s.feature_256_1024 for s in sample_list])  # (N, 256, 1024)
            embeddings = embeddings.to(device)

            # 计算与 query_sample 的余弦相似度
            similarities = F.cosine_similarity(embeddings, query_embedding.unsqueeze(0), dim=-1)  # (N, 256)
            mean_similarities = similarities.mean(dim=-1)  # 每个样本与 query_sample 的平均相似度 (N,)
            
            # 找到最相似的样本
            most_similar_index = torch.argmin(mean_similarities).cpu().item()
            target_sample = sample_list[most_similar_index]
        if self.args.update_strategy == "gradient_prototype":
            self.update_based_on_gradient_and_prototype(target_sample,query_sample)
        elif self.args.update_strategy == "fixed":
            self.update_based_on_fixed(target_sample,query_sample,self.args.alpha)
        elif self.args.update_strategy == "cyclic":
            self.update_based_on_cyclic_momentum(target_sample,query_sample)
        elif self.args.update_strategy == "multi_step":
            self.update_based_on_multi_step(target_sample,query_sample)
        elif self.args.update_strategy == "joint_rate":
            self.update_based_on_joint_rate(target_sample,query_sample)
        else:
            print("update_strategy is not effective.")
            return

    def adjust_learning_rate(self, base_lr, timestep):
        """
        根据时间步衰减调整学习率
        base_lr: 初始学习率
        timestep: 当前时间步
        """
        # 定义衰减因子 beta，可以在 args 中设置
        decay_factor = self.args.decay
        # 计算时间步衰减后的学习率
        adjusted_lr = base_lr / (1 + decay_factor * timestep)
        return adjusted_lr

    def compute_update_rate(self, sample,alpha = 0.5):
        clip_similairity = (sample.quality+1)/2
        update_rate = alpha * clip_similairity + (1-alpha) * sample.margin

        return update_rate
    
    def compute_gradient(self, sample,alpha = 0.2,beta = 0,delta=0.2):
        error_rate = sum(self.error_history[sample.label]) / len(self.error_history[sample.label]) if len(self.error_history[sample.label]) > 0 else 0

        if len(self.error_history[sample.label]) == 10:
            if sample.label not in self.error_rate:
                self.error_rate[sample.label] = []
            self.error_rate[sample.label].append(error_rate)

        clip_similairity = (sample.quality+1)/2

        margin = sample.margin
        support_gradient = alpha * clip_similairity + beta * margin+ delta * error_rate
        self.class_gradients[sample.label].append(support_gradient)

        return support_gradient
    
    def update_based_on_gradient_and_prototype(self,target_sample,query_sample):  #  alpha = 0.4,beta = 0.5,delta=0.1 :63.22
        label = query_sample.label
        inference_result = 1 if query_sample.pseudo_label == label else 0
        # 更新该类别的推理历史记录
        self.error_history[label].append(1 - inference_result)  # 记录错误推理
        # 计算 Support Gradient
        support_gradient = self.compute_gradient(query_sample,self.args.alpha,self.args.beta,self.args.delta)
        # 获取当前类别的样本列表
        sample_list = self.label2sample[label]

        target_sample.feature_256_1024 = (1 - support_gradient) * target_sample.feature_256_1024 + support_gradient * query_sample.feature_256_1024
        # 更新类别原型
        self.label_to_prototype[label] = torch.mean(torch.stack([s.feature_256_1024 for s in sample_list]), dim=0)

    def update_based_on_joint_rate(self,target_sample,query_sample): 
        device_set = "cuda:" + str(self.args.device)
        device = torch.device(device_set)
        query_embed = query_sample.feature_256_1024.to(device)
        label = query_sample.label

        # 计算 update rate 
        update_rate = self.compute_update_rate(query_sample,self.args.alpha)

        # 计算 保留率
        current_prototype = self.label_to_prototype[label].to(device)
        similarity_to_prototype = F.cosine_similarity(query_embed, current_prototype, dim=-1)
        similarity_to_prototype = similarity_to_prototype.mean(dim=-1).cpu().item()
        maintain_rate = (similarity_to_prototype+1)/2

        # 获取当前类别的样本列表
        sample_list = self.label2sample[label]

        target_sample.feature_256_1024 = maintain_rate * target_sample.feature_256_1024 + update_rate * query_sample.feature_256_1024
        
        # 更新类别原型
        self.label_to_prototype[label] = torch.mean(torch.stack([s.feature_256_1024 for s in sample_list]), dim=0)

    def update_based_on_fixed(self,target_sample,query_sample,alpha=0.2):   
         # 获取当前类别的样本列表
        label = query_sample.label
        sample_list = self.label2sample[label]

        target_sample.feature_256_1024 = (1-alpha) * target_sample.feature_256_1024 + alpha * query_sample.feature_256_1024

        # 更新类别原型
        self.label_to_prototype[query_sample.label] = torch.mean(torch.stack([s.feature_256_1024 for s in sample_list]), dim=0)
    
    def update_based_on_cyclic_momentum(self, target_sample,query_sample):
        sample_list = self.label2sample[query_sample.label]
        current_lr = self.compute_cyclic_beta(self.timestep,cycle_length=self.args.M)
        
        target_sample.feature_256_1024 = (1 - current_lr) * target_sample.feature_256_1024 + current_lr * query_sample.feature_256_1024

        # 更新类别原型
        self.label_to_prototype[query_sample.label] = torch.mean(torch.stack([s.feature_256_1024 for s in sample_list]), dim=0)

        # 更新时间步
        self.timestep += 1

    def compute_cyclic_beta(self, timestep, beta_max=0.9, beta_min=0.1, cycle_length=1000):
        cycle_position = timestep % cycle_length
        beta_t = beta_min + 0.5 * (beta_max - beta_min) * (1 + math.cos(2 * math.pi * cycle_position / cycle_length))
        return beta_t

    def update_based_on_multi_step(self, target_sample,query_sample,num_steps=4, delta=0.05):
        sample_list = self.label2sample[query_sample.label]
        for _ in range(num_steps):
            target_sample.feature_256_1024 = target_sample.feature_256_1024 + delta * (query_sample.feature_256_1024 - target_sample.feature_256_1024)
        
        # 更新类别原型
        self.label_to_prototype[query_sample.label] = torch.mean(torch.stack([s.feature_256_1024 for s in sample_list]), dim=0)
