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
        # error_rate 是一个类 ID 到 error rate list 的映射
        self.error_rate = {}
        self.class_gradients = defaultdict(list)
        self.timestep = 0
        self.match_pool = []
        self.dismatch_pool = []

    def get_final_query(self, sample):
        demonstrations = self.get_demonstrations_from_bank(sample)
        ice_text = ""
        ice_img=[]
        if demonstrations is not None:
            for dm in demonstrations:
                ice_img.append(dm.image)
                ice_text += get_imagenet_prompt(dm.label)
                    
        ice_img.append(sample.image)
        ice_text += get_imagenet_prompt()
        return ice_img,ice_text,demonstrations

    def get_demonstrations_from_bank(self, sample):
        if self.args.dnum == 0:
            return []
        if self.args.select_strategy == "random":
            indices = self.get_random(sample)
        elif self.args.select_strategy == "cosine":
            indices = self.get_topk_cosine(sample)
        elif self.args.select_strategy == "l2":
            indices = self.get_topk_euclidean(sample)
        else:
            print("select_strategy is not effective.")
            return
        return [self.demonstrations[i] for i in indices]

    def get_random(self, sample):
        indices = random.sample(range(len(self.demonstrations)), self.args.dnum)
        return indices

    def get_topk_cosine(self, sample):
        demonstration_embeds = torch.stack([sample.embed for sample in self.demonstrations], dim=0)
        device_set = "cuda:" + str(self.args.device)
        device = torch.device(device_set)
        demonstration_embeds = demonstration_embeds.to(device)
        sample_embed = sample.embed.to(device)
        scores = torch.cosine_similarity(demonstration_embeds, sample_embed.unsqueeze(0), dim=-1)
        #mean_scores = scores.mean(dim=1)  # 现在的形状是 (N,)
        values, indices = torch.topk(scores, self.args.dnum, largest=True)
        indices = indices.cpu().tolist()
        return indices
    
    def get_topk_euclidean(self, sample):
        device_set = "cuda:" + str(self.args.device)
        device = torch.device(device_set)
        # 将所有 demonstrations 的嵌入拼接成一个张量
        demonstration_embeds = torch.stack([sample.embed for sample in self.demonstrations], dim=0)
        
        demonstration_embeds = demonstration_embeds.to(device)
        
        # 将样本的嵌入也移动到相同的设备
        sample_embed = sample.embed.to(device)
        
        # 计算 sample_embed 和 demonstration_embeds 的欧几里得距离
        distances = F.pairwise_distance(demonstration_embeds, sample_embed.unsqueeze(0), p=2)
        
        # 选取距离最小的 k 个样本
        values, indices = torch.topk(distances, self.args.dnum, largest=False)  # 使用 smallest k 的方式
        
        # 将索引从 GPU 移动到 CPU，并转换为列表返回
        indices = indices.cpu().tolist()
        return indices
    
    def update(self):
        if len(self.pool) == 0:
            return
        samples_to_remove = self.pool[:1]
        self.pool = self.pool[1:]
        if self.args.dataset_mode == "balanced":
            if self.args.update_strategy == "prototype":
                self.update_based_on_prototype(samples_to_remove[0])
            elif self.args.update_strategy == "minmargin":
                self.update_based_on_minmargin(samples_to_remove[0])
            elif self.args.update_strategy == "maxmargin":
                self.update_based_on_maxmargin(samples_to_remove[0])
            else:
                print(f"{self.args.update_strategy} is not effective.")
                return
        else: # imbalanced
            if self.args.update_strategy == "prototype":
                self.update_based_on_balance_prototype(samples_to_remove[0],self.max_samples_num)
            else:
                print("update_strategy is not effective.")
                return
        
    def update_based_on_prototype(self,sample_to_remove): # 58.68
        query_embed = sample_to_remove.embed
        label = sample_to_remove.label

        sample_list = self.label2sample[label]
        embed_list = [sample.embed for sample in sample_list]

        prototype = torch.mean(torch.stack(embed_list), dim=0)

        query_similarity = torch.cosine_similarity(query_embed.unsqueeze(0), prototype.unsqueeze(0)).item()

        similarities = torch.cosine_similarity(torch.stack(embed_list), prototype.unsqueeze(0))

        least_similar_index = torch.argmin(similarities).item()

        # 判断是否需要替换
        if query_similarity > similarities[least_similar_index]:
            self.demonstrations.remove(sample_list[least_similar_index])
            self.demonstrations.append(sample_to_remove)
            self.label2sample[label].remove(sample_list[least_similar_index])
            self.label2sample[label].append(sample_to_remove)
            
        assert len(self.demonstrations) == self.args.M
    
    def update_based_on_balance_prototype(self, sample,max_samples_num):
            label = sample.label
            query_embed = sample.embed

            # 类别不在 memory bank 中，直接加入
            if label not in self.label2sample:
                self.demonstrations.append(sample)
                self.label2sample[label] = [sample]
        
            elif len(self.label2sample[label]) < max_samples_num:
                self.demonstrations.append(sample)
                self.label2sample[label].append(sample)
        
            else:
                # 找到类内与prototype最不相似的样本
                sample_list = self.label2sample[label]
                embed_list = [s.embed for s in sample_list]

                prototype = torch.mean(torch.stack(embed_list), dim=0)
                query_similarity = torch.cosine_similarity(query_embed.unsqueeze(0), prototype.unsqueeze(0)).item()
                similarities = torch.cosine_similarity(torch.stack(embed_list), prototype.unsqueeze(0))
                least_similar_index = torch.argmin(similarities).item()

                if query_similarity > similarities[least_similar_index]:
                    removed_sample = sample_list[least_similar_index]
                    self.label2sample[label].remove(removed_sample)
                    self.demonstrations.remove(removed_sample)
                    self.demonstrations.append(sample)
                    self.label2sample[label].append(sample)

            while len(self.demonstrations) > self.args.M:
                # 从最大类别中删掉一个离prototype最远的样本
                max_label = max(self.label2sample, key=lambda k: len(self.label2sample[k]))
                max_sample_list = self.label2sample[max_label]
                removed_sample,_ = self.get_least_similar_sample(max_sample_list)
                self.label2sample[max_label].remove(removed_sample)
                self.demonstrations.remove(removed_sample)

    def get_least_similar_sample(self, sample_list):
        """计算 prototype 并返回与其最不相似的样本"""
        embed_list = [sample.embed for sample in sample_list]
        prototype = torch.mean(torch.stack(embed_list), dim=0)
        similarities = torch.cosine_similarity(torch.stack(embed_list), prototype.unsqueeze(0))
        least_similar_index = torch.argmin(similarities).item()
        return sample_list[least_similar_index], similarities[least_similar_index].item()

    def update_online(self,query_sample):
        if self.args.dataset_mode == "balanced":
            device_set = "cuda:" + str(self.args.device)
            device = torch.device(device_set)
            label = query_sample.label
            
            # 获取当前类别的样本列表
            sample_list = self.label2sample[label]
            # 获取当前类别的原型向量
            current_prototype = self.label_to_prototype[label] #(256,1024)
            current_prototype = current_prototype.to(device)
        
            embeddings = torch.stack([s.embed for s in sample_list])
            embeddings = embeddings.to(device)

            similarities = F.cosine_similarity(embeddings, current_prototype.unsqueeze(0), dim=-1)
           
            least_similar_index = torch.argmin(similarities).cpu().item()

            target_sample = sample_list[least_similar_index]

        if self.args.update_strategy == "gradient_prototype":
            self.update_based_on_gradient_and_prototype(query_sample)
        elif self.args.update_strategy == "time_decay":
            self.update_prototype_with_time_decay(query_sample)
        elif self.args.update_strategy == "fixed":
            self.update_based_on_fixed(target_sample,query_sample)
        elif self.args.update_strategy == "cyclic":
            self.update_based_on_cyclic_momentum(target_sample,query_sample)
        elif self.args.update_strategy == "multi_step":
            self.update_based_on_multi_step(target_sample,query_sample)
        elif self.args.update_strategy == "new":
            self.update_based_on_new(query_sample)
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
    
    def update_based_on_gradient_and_prototype(self,query_sample):  #  alpha = 0.4,beta = 0.5,delta=0.1 :63.22
        query_embed = query_sample.embed
        label = query_sample.label
        inference_result = 1 if query_sample.pseudo_label == label else 0
        # 更新该类别的推理历史记录
        self.error_history[label].append(1 - inference_result)  # 记录错误推理
        # 计算 Support Gradient
        support_gradient = self.compute_gradient(query_sample,self.args.alpha,self.args.beta,self.args.delta)
        self.support_gradient_list.append(support_gradient)
        
        # 获取当前类别的样本列表
        sample_list = self.label2sample[label]
        # 获取当前类别的原型向量
        current_prototype = self.label_to_prototype[label]

        # 找到当前类别中最不相似的样本（与原型相距最远的样本）
        similarities = torch.cosine_similarity(torch.stack([s.embed for s in sample_list]), current_prototype.unsqueeze(0))
        least_similar_index = torch.argmin(similarities).item()

        least_similar_sample = sample_list[least_similar_index]
        least_similar_sample.embed = (1 - support_gradient) * least_similar_sample.embed + support_gradient * query_embed
        # 更新类别原型
        self.label_to_prototype[label] = torch.mean(torch.stack([s.embed for s in sample_list]), dim=0)
            
        assert len(self.demonstrations) == self.args.M

    def update_prototype_with_time_decay(self, query_sample): 
        """
        基于时间步衰减和原型反馈更新样本嵌入向量
        """
        # 获取 query 的嵌入向量和标签
        query_embed = query_sample.embed
        label = query_sample.label
        inference_result = 1 if query_sample.pseudo_label == label else 0
        # 更新该类别的推理历史记录
        self.error_history[label].append(1 - inference_result)  # 记录错误推理

        # 计算 Support Gradient
        support_gradient = self.compute_gradient(query_sample, self.args.alpha, self.args.beta, self.args.delta)
        self.support_gradient_list.append(support_gradient)

        # 动态调整学习率（根据时间步）
        base_lr = support_gradient  # 使用计算的 gradient 作为 base_lr
        current_lr = self.adjust_learning_rate(base_lr, self.timestep)

        # 获取当前类别的样本列表
        sample_list = self.label2sample[label]
        # 获取当前类别的原型向量
        current_prototype = self.label_to_prototype[label]

        # 计算与原型的相似度，找到最不相似的样本
        similarities = torch.cosine_similarity(torch.stack([s.embed for s in sample_list]), current_prototype.unsqueeze(0))
        least_similar_index = torch.argmin(similarities).item()
        target_sample = sample_list[least_similar_index]

        # 更新目标样本的嵌入向量，使用当前学习率
        target_sample.embed = (1 - current_lr) * target_sample.embed + current_lr * query_embed

        # 更新类别原型
        self.label_to_prototype[label] = torch.mean(torch.stack([s.embed for s in sample_list]), dim=0)

        # 更新时间步
        self.timestep += 1
    
    def update_based_on_fixed(self,target_sample,query_sample,gradient=0.2):   
         # 获取当前类别的样本列表
        label = query_sample.label
        sample_list = self.label2sample[label]

        target_sample.embed = (1 - gradient) * target_sample.embed + gradient * query_sample.embed
        #target_sample.feature_256_1024 = (1 - gradient) * target_sample.feature_256_1024 + gradient * query_sample.feature_256_1024
    
        # 更新类别原型
        self.label_to_prototype[query_sample.label] = torch.mean(torch.stack([s.embed for s in sample_list]), dim=0)
        
    def update_based_on_cyclic_momentum(self, target_sample,query_sample):
        sample_list = self.label2sample[query_sample.label]
        current_lr = self.compute_cyclic_beta(self.timestep,cycle_length=self.args.M)
        
        target_sample.embed = (1 - current_lr) * target_sample.embed + current_lr * query_sample.embed

        # 更新类别原型
        self.label_to_prototype[query_sample.label] = torch.mean(torch.stack([s.embed for s in sample_list]), dim=0)

        # 更新时间步
        self.timestep += 1

    def compute_cyclic_beta(self, timestep, beta_max=0.8, beta_min=0.2, cycle_length=1000):
        cycle_position = timestep % cycle_length
        beta_t = beta_min + 0.5 * (beta_max - beta_min) * (1 + math.cos(2 * math.pi * cycle_position / cycle_length))
        return beta_t

    def update_based_on_multi_step(self, target_sample,query_sample,num_steps=5, delta=0.1):
        sample_list = self.label2sample[query_sample.label]
        for _ in range(num_steps):
            target_sample.embed = target_sample.embed + delta * (query_sample.embed - target_sample.embed)
        
        # 更新类别原型
        self.label_to_prototype[query_sample.label] = torch.mean(torch.stack([s.embed for s in sample_list]), dim=0)

    
    def update_based_on_new(self,query_sample):
        updated_labels = set()
        # self.dismatch_pool中的样本是一定要更新的，把embed靠近所属类的prototype
        for d in self.dismatch_pool:
            p = self.label_to_prototype[d.label]

            # 靠近p
            d.embed = 0.9 * d.embed + 0.1 * p
            updated_labels.add(d.label)  # 记录该样本的类别标签，表示其 prototype 需要更新

        # 对于self.match_pool中的样本，如果预测结果对了，不需要更新
        if query_sample.label == query_sample.pseudo_label:
            pass
        else: # 否则用数据流更新match pool 中的样本
            for m in self.match_pool:
                m.embed  =0.9*m.embed+ 0.1*query_sample.embed
                updated_labels.add(m.label)

        # 更新已记录的每个类别标签的 prototype
        for label in updated_labels:
            # 获取当前类别的所有样本的 embed
            sample_embeds = [s.embed for s in self.label2sample[label]]
            # 计算新的 prototype 为该类别中所有样本的平均 embed
            self.label_to_prototype[label] = torch.mean(torch.stack(sample_embeds), dim=0)

        # 更新结束后，两个pool 清空
        self.match_pool = []
        self.dismatch_pool = []