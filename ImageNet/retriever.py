import torch
import random
from utils import get_imagenet_prompt
import os
import json
import numpy as np
import time
from collections import defaultdict, deque
import logging
import math
logger = logging.getLogger(__name__)

class DynamicReteiever:
    def __init__(self, args):
        self.args = args
        self.demonstrations = []
        self.label2sample = dict()
        self.dnum = 4
        self.label_to_prototype = {}
        self.error_history = defaultdict(lambda: deque(maxlen=10))
        self.support_gradient_list = []
        # error_rate 是一个类 ID 到 error rate list 的映射
        self.error_rate = {}
        self.class_gradients = defaultdict(list)
        self.timestep = 0
    
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
        if self.args.select_strategy == "random":
            indices = self.get_random(sample)
        elif self.args.select_strategy == "cosine":
            indices = self.get_topk_cosine(sample)
        elif self.args.select_strategy == "l2":
            indices = self.get_topk_l2(sample)
        else:
            print("select_strategy is not effective.")
            return
        return [self.demonstrations[i] for i in indices]

    def get_random(self, sample):
        indices = random.sample(range(len(self.demonstrations)), self.dnum)
        return indices

    def get_topk_cosine(self, sample):
        demonstration_embeds = torch.stack([sample.embed for sample in self.demonstrations], dim=0)
        #logging.info(f"demonstration_embeds 的设备在: {demonstration_embeds.device}")
        device_set = "cuda:" + str(self.args.device)
        device = torch.device(device_set)
        demonstration_embeds = demonstration_embeds.to(device)
        sample_embed = sample.embed.to(device)
        #scores = torch.cosine_similarity(demonstration_embeds, sample.embed, dim=-1)
        scores = torch.cosine_similarity(demonstration_embeds, sample_embed, dim=-1)
        values, indices = torch.topk(scores, self.dnum, largest=True)
        indices = indices.cpu().tolist()
        #indices = indices.tolist()
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
        if self.args.update_strategy == "gradient_prototype":
            self.update_based_on_gradient_and_prototype(query_sample)
        elif self.args.update_strategy == "gradient_prototype":
            self.update_based_on_gradient_and_prototype(query_sample)
        elif self.args.update_strategy == "time_decay":
            self.update_prototype_with_time_decay(query_sample)
        elif self.args.update_strategy == "fixed_gradient":
            self.update_based_on_fixed_gradient(query_sample,gradient =self.args.gradient)
        elif self.args.update_strategy == "fixed_gradient_time_decay":  
            self.update_based_on_fixed_gradient_and_time_decay(query_sample,gradient =self.args.gradient)
        elif self.args.update_strategy == "cyclic_momentum_quality":  
            self.update_based_on_cyclic_momentum_quality(query_sample)
        elif self.args.update_strategy == "cyclic_momentum":  
            self.update_based_on_cyclic_momentum(query_sample)
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

    def compute_gradient(self, sample,alpha = 0,beta = 1.0,delta=0):
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

    def compute_cyclic_beta_quality(self, timestep, sample_quality, beta_max=0.9, beta_min=0.1, cycle_length=1000):
        """
        计算周期性动量系数 beta 随着时间步周期性变化
        :param timestep: 当前推理的时间步
        :param sample_quality: 样本质量（假设是 0 到 1 之间的浮点数）
        :param beta_max: 动量的最大值
        :param beta_min: 动量的最小值
        :param cycle_length: 动量变化的周期长度
        :return: 动态调整后的 beta 值
        """
        cycle_position = timestep % cycle_length
        beta_t = beta_min + 0.5 * (beta_max - beta_min) * (1 + math.cos(2 * math.pi * cycle_position / cycle_length))
        return beta_t * sample_quality

    def compute_cyclic_beta(self, timestep, beta_max=0.9, beta_min=0.1, cycle_length=1000):
        cycle_position = timestep % cycle_length
        beta_t = beta_min + 0.5 * (beta_max - beta_min) * (1 + math.cos(2 * math.pi * cycle_position / cycle_length))
        return beta_t
    
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
    
    def update_based_on_fixed_gradient(self,query_sample,gradient): 
        query_embed = query_sample.embed
        label = query_sample.label
        # 计算 Support Gradient
        support_gradient = gradient
        
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
            
    
    def update_based_on_fixed_gradient_and_time_decay(self, query_sample,gradient):
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
        support_gradient = gradient

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

    def update_based_on_cyclic_momentum_quality(self, query_sample):
        query_embed = query_sample.embed
        label = query_sample.label
        inference_result = 1 if query_sample.pseudo_label == label else 0
        # 更新该类别的推理历史记录
        self.error_history[label].append(1 - inference_result)  # 记录错误推理

        # 获取当前类别的样本列表
        sample_list = self.label2sample[label]
        # 获取当前类别的原型向量
        current_prototype = self.label_to_prototype[label]

        # 计算与原型的相似度，找到最不相似的样本
        similarities = torch.cosine_similarity(torch.stack([s.embed for s in sample_list]), current_prototype.unsqueeze(0))
        least_similar_index = torch.argmin(similarities).item()
        target_sample = sample_list[least_similar_index]

        current_lr = self.compute_cyclic_beta_quality(self.timestep,query_sample.quality)
        
        target_sample.embed = (1 - current_lr) * target_sample.embed + current_lr * query_embed

        # 更新类别原型
        self.label_to_prototype[label] = torch.mean(torch.stack([s.embed for s in sample_list]), dim=0)

        # 更新时间步
        self.timestep += 1

    def update_based_on_cyclic_momentum(self, query_sample):
        query_embed = query_sample.embed
        label = query_sample.label
        inference_result = 1 if query_sample.pseudo_label == label else 0
        # 更新该类别的推理历史记录
        self.error_history[label].append(1 - inference_result)  # 记录错误推理

        # 获取当前类别的样本列表
        sample_list = self.label2sample[label]
        # 获取当前类别的原型向量
        current_prototype = self.label_to_prototype[label]

        # 计算与原型的相似度，找到最不相似的样本
        similarities = torch.cosine_similarity(torch.stack([s.embed for s in sample_list]), current_prototype.unsqueeze(0))
        least_similar_index = torch.argmin(similarities).item()
        target_sample = sample_list[least_similar_index]

        current_lr = self.compute_cyclic_beta(self.timestep)
        
        target_sample.embed = (1 - current_lr) * target_sample.embed + current_lr * query_embed

        # 更新类别原型
        self.label_to_prototype[label] = torch.mean(torch.stack([s.embed for s in sample_list]), dim=0)

        # 更新时间步
        self.timestep += 1

    def update_based_on_balance_balance_gradient_prototype(self, query_sample,max_samples_num=10): # 待评估
        query_embed = query_sample.embed
        label = query_sample.label

        if label not in self.label2sample:
            self.demonstrations.append(query_sample)
            self.label2sample[label] = [query_sample]
        elif len(self.label2sample[label]) < max_samples_num:
            self.demonstrations.append(query_sample)
            self.label2sample[label].append(query_sample)
        else:
            inference_result = 1 if query_sample.pseudo_label == label else 0
            # 更新该类别的推理历史记录
            self.error_history[label].append(1 - inference_result)  # 记录错误推理

            # 计算 Support Gradient
            support_gradient = self.compute_gradient(query_sample)
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
        self.label_to_prototype[label] = torch.mean(torch.stack([s.embed for s in self.label2sample[label]]), dim=0)
        
        while len(self.demonstrations) > self.args.M:
            # 从最大类别中删掉一个离prototype最远的样本
            max_label = max(self.label2sample, key=lambda k: len(self.label2sample[k]))
            max_sample_list = self.label2sample[max_label]
            removed_sample,_ = self.get_least_similar_sample(max_sample_list)
            self.label2sample[max_label].remove(removed_sample)
            self.demonstrations.remove(removed_sample)
            # 更新最大类别的prototype
            self.label_to_prototype[label] = torch.mean(torch.stack([s.embed for s in self.label2sample[max_label]]), dim=0)
        
        assert len(self.demonstrations) == self.args.M


        
    


        