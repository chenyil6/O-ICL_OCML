import torch
import random
from utils import get_imagenet_prompt
import os
import json
import numpy as np
import time
from collections import defaultdict, deque

class DynamicReteiever:
    def __init__(self, args):
        self.args = args
        self.demonstrations = []
        self.pool = []
        self.label2sample = dict()
        self.dnum = 4
        self.label_to_prototype = {}
        self.error_history = defaultdict(lambda: deque(maxlen=10))
        self.support_gradient_list = []
    
    def get_final_query(self, sample):
        demonstrations = self.get_demonstrations_from_bank(sample)
        ice_text = ""
        ice_img=[]
        if demonstrations is not None:
            for dm in demonstrations:
                ice_img.append(dm.image)
                ice_text += get_imagenet_prompt(dm.pseudo_label if dm.pseudo_label is not None else dm.label)

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
        scores = torch.cosine_similarity(demonstration_embeds, sample.embed, dim=-1)
        values, indices = torch.topk(scores, self.dnum, largest=True)
        indices = indices.tolist()
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
    
    def update_based_on_minmargin(self, sample_to_remove): # 51.84
        query_embed = sample_to_remove.embed
        label = sample_to_remove.label
        query_marigin = self.compute_minMargin(query_embed,label)
        sample_list = self.label2sample[label]

        margins = [self.compute_minMargin(s.embed, label) for s in sample_list]
        
        # 找到 margin 最小的样本
        min_margin_index = torch.argmin(torch.tensor(margins)).item()

        if query_marigin > margins[min_margin_index]:
            self.demonstrations.remove(sample_list[min_margin_index])
            self.demonstrations.append(sample_to_remove)
            self.label2sample[label].remove(sample_list[min_margin_index])
            self.label2sample[label].append(sample_to_remove)

            self.label_to_prototype[label] = torch.mean(torch.stack([s.embed for s in self.label2sample[label]]), dim=0)
            
        assert len(self.demonstrations) == self.args.M
    
    def update_based_on_maxmargin(self, sample_to_remove): # 55.56
        query_embed = sample_to_remove.embed
        label = sample_to_remove.label
        query_marigin = self.compute_maxMargin(query_embed,label)
        sample_list = self.label2sample[label]

        margins = [self.compute_maxMargin(s.embed, label) for s in sample_list]
        
        # 找到 margin 最小的样本
        min_margin_index = torch.argmin(torch.tensor(margins)).item()

        if query_marigin > margins[min_margin_index]:
            self.demonstrations.remove(sample_list[min_margin_index])
            self.demonstrations.append(sample_to_remove)
            self.label2sample[label].remove(sample_list[min_margin_index])
            self.label2sample[label].append(sample_to_remove)

            self.label_to_prototype[label] = torch.mean(torch.stack([s.embed for s in self.label2sample[label]]), dim=0)
            
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

    def replace_sample(self, old_sample, new_sample, label):
        """替换 memory bank 中的旧样本"""
        self.demonstrations.remove(old_sample)
        self.label2sample[label].remove(old_sample)
        self.demonstrations.append(new_sample)
        self.label2sample[label].append(new_sample)
    
    def update_online(self,query_sample):
        if self.args.update_strategy == "default_prototype":
            self.update_based_on_default_prototype(query_sample)
        elif self.args.update_strategy == "default_minmargin":
            self.update_based_on_default_minmargin(query_sample)
        elif self.args.update_strategy == "default_maxmargin":
            self.update_based_on_default_maxmargin(query_sample)
        elif self.args.update_strategy == "gradient_prototype":
            self.update_based_on_gradient_and_prototype(query_sample)
        elif self.args.update_strategy == "gradient_maxmargin":
            self.update_based_on_gradient_and_maxmargin(query_sample)
        elif self.args.update_strategy == "gradient_minmargin":
            self.update_based_on_gradient_and_minmargin(query_sample)
        elif self.args.update_strategy == "gradient_minmargin_topk":
            self.update_based_on_gradient_and_minmargin_topk(query_sample)
        elif self.args.update_strategy == "gradient_equal_1":
            self.update_based_on_gradient_equal_1(query_sample)
        elif self.args.update_strategy == "maxmargin_equal_1":
            self.update_based_on_gradient_and_maxmargin_euqal_1(query_sample)
        else:
            print("update_strategy is not effective.")
            return

    def compute_gradient(self, sample):
        alpha = 0.5
        beta = 0.1
        delta = 0.4

        error_rate = sum(self.error_history[sample.label]) / len(self.error_history[sample.label]) if len(self.error_history[sample.label]) > 0 else 0

        clip_similairity = (sample.similarity+1)/2

        confidence = sample.gt_score
        # 基础 Support Gradient 公式
        support_gradient = alpha * (1 - confidence) + beta *error_rate  + delta * clip_similairity

        return support_gradient

    def compute_maxMargin(self, embed, class_label):
        """
        计算样本的 margin (类内相似度 - 类间相似度)
        embed: 样本的嵌入
        class_label: 当前样本的类别
        """
        # 类内相似度
        prototype_same_class = self.label_to_prototype[class_label]
        similarity_intra = torch.cosine_similarity(embed, prototype_same_class, dim=0)

        # 类间相似度 (与其他类别的原型的最小相似度)
        similarities_inter = []
        for label, prototype in self.label_to_prototype.items():
            if label != class_label:
                similarity_inter = torch.cosine_similarity(embed, prototype, dim=0)
                similarities_inter.append(similarity_inter)
        
        # 取类间最小相似度
        min_similarity_inter = min(similarities_inter) if similarities_inter else torch.tensor(-float('inf'))

        # 计算 margin
        margin = similarity_intra - min_similarity_inter
        return margin
    
    def compute_minMargin(self, embed, class_label):
        """
        计算样本的 margin (类内相似度 - 类间相似度)
        embed: 样本的嵌入
        class_label: 当前样本的类别
        """
        # 类内相似度
        prototype_same_class = self.label_to_prototype[class_label]
        similarity_intra = torch.cosine_similarity(embed, prototype_same_class, dim=0)

        # 类间相似度 (与其他类别的原型的最大相似度)
        similarities_inter = []
        for label, prototype in self.label_to_prototype.items():
            if label != class_label:
                similarity_inter = torch.cosine_similarity(embed, prototype, dim=0)
                similarities_inter.append(similarity_inter)
        
        # 取类间最大相似度
        max_similarity_inter = max(similarities_inter) if similarities_inter else torch.tensor(-float('inf'))

        # 计算 margin
        margin = similarity_intra - max_similarity_inter
        return margin

    def compute_minMargin_topk(self, embed, class_label,k=5):
        """
        计算样本的 margin (类内相似度 - 类间相似度)
        embed: 样本的嵌入
        class_label: 当前样本的类别
        """
        # 类内相似度
        prototype_same_class = self.label_to_prototype[class_label]
        similarity_intra = torch.cosine_similarity(embed, prototype_same_class, dim=0)

        # 类间相似度 (与其他类别的原型的最小相似度)
        similarities_inter = []
        for label, prototype in self.label_to_prototype.items():
            if label != class_label:
                similarity_inter = torch.cosine_similarity(embed, prototype, dim=0)
                similarities_inter.append(similarity_inter)
        
        # 选择 k 个最相似的负样本
        similarities_inter = sorted(similarities_inter, reverse=True)  # 从大到小排序
        top_k_similarities_inter = similarities_inter[:k]

        # 计算 k 个负样本的平均相似度
        avg_similarity_inter = sum(top_k_similarities_inter) / k if top_k_similarities_inter else torch.tensor(-float('inf'))

        # 计算 margin
        margin = similarity_intra - avg_similarity_inter
        return margin
    
    def update_based_on_default_prototype(self,query_sample):  #  52.16
        query_embed = query_sample.embed
        label = query_sample.label
        inference_result = 1 if query_sample.pseudo_label == label else 0
        confidence = query_sample.gt_score
        # 更新该类别的推理历史记录
        self.error_history[label].append(1 - inference_result)  # 记录错误推理

        # 计算 Support Gradient
        support_gradient = self.compute_gradient(query_sample)
        self.support_gradient_list.append(support_gradient)
        # 获取当前类别的原型向量
        current_prototype = self.label_to_prototype[label]

        query_similarity = torch.cosine_similarity(query_embed.unsqueeze(0), current_prototype.unsqueeze(0)).item()

        # 找到当前类别中最不相似的样本（与原型相距最远的样本）
        sample_list = self.label2sample[label]
        similarities = torch.cosine_similarity(torch.stack([s.embed for s in sample_list]), current_prototype.unsqueeze(0))
        least_similar_index = torch.argmin(similarities).item()

        if inference_result == 1:
            least_similar_sample = sample_list[least_similar_index]
            least_similar_sample.embed = (1 - support_gradient) * least_similar_sample.embed + support_gradient * query_embed
            # 更新类别原型
            self.label_to_prototype[label] = torch.mean(torch.stack([s.embed for s in sample_list]), dim=0)
        else: # 如果判断错误，那么一定要进行替换判断
            if query_similarity > similarities[least_similar_index]:
                self.demonstrations.remove(sample_list[least_similar_index])
                self.demonstrations.append(query_sample)
                self.label2sample[label].remove(sample_list[least_similar_index])
                self.label2sample[label].append(query_sample)
                # 更新类别原型
                self.label_to_prototype[label] = torch.mean(torch.stack([s.embed for s in self.label2sample[label]]), dim=0)
        assert len(self.demonstrations) == self.args.M
    
    def update_based_on_default_minmargin(self, query_sample): # 待评估
        query_embed = query_sample.embed
        label = query_sample.label
        inference_result = 1 if query_sample.pseudo_label == label else 0
        # 更新该类别的推理历史记录
        self.error_history[label].append(1 - inference_result)  # 记录错误推理

        # 计算 Support Gradient
        support_gradient = self.compute_gradient(query_sample)
        self.support_gradient_list.append(support_gradient)
        
        query_minmargin = self.compute_minMargin(query_embed,label)
        
        # 找到当前类别中最不相似的样本（与原型相距最远的样本）
        sample_list = self.label2sample[label]
        # 计算支持集中每个样本的 margin
        margins = [self.compute_minMargin(s.embed, label) for s in sample_list]

        # 找到 margin 最小的样本
        min_margin_index = torch.argmin(torch.tensor(margins)).item()

        if inference_result == 1:
            least_similar_sample = sample_list[min_margin_index]
            least_similar_sample.embed = (1 - support_gradient) * least_similar_sample.embed + support_gradient * query_embed
            # 更新类别原型
            self.label_to_prototype[label] = torch.mean(torch.stack([s.embed for s in sample_list]), dim=0)
        else: # 如果判断错误，那么一定要进行替换判断
            if query_minmargin > margins[min_margin_index]:
                self.demonstrations.remove(sample_list[min_margin_index])
                self.demonstrations.append(query_sample)
                self.label2sample[label].remove(sample_list[min_margin_index])
                self.label2sample[label].append(query_sample)
                # 更新类别原型
                self.label_to_prototype[label] = torch.mean(torch.stack([s.embed for s in self.label2sample[label]]), dim=0)
        assert len(self.demonstrations) == self.args.M

    def update_based_on_default_maxmargin(self, query_sample): # 待评估
        query_embed = query_sample.embed
        label = query_sample.label
        inference_result = 1 if query_sample.pseudo_label == label else 0
        # 更新该类别的推理历史记录
        self.error_history[label].append(1 - inference_result)  # 记录错误推理

        # 计算 Support Gradient
        support_gradient = self.compute_gradient(query_sample)
        self.support_gradient_list.append(support_gradient)
        
        query_minmargin = self.compute_maxMargin(query_embed,label)
        
        # 找到当前类别中最不相似的样本（与原型相距最远的样本）
        sample_list = self.label2sample[label]
        # 计算支持集中每个样本的 margin
        margins = [self.compute_maxMargin(s.embed, label) for s in sample_list]

        # 找到 margin 最小的样本
        min_margin_index = torch.argmin(torch.tensor(margins)).item()

        if inference_result == 1:
            least_similar_sample = sample_list[min_margin_index]
            least_similar_sample.embed = (1 - support_gradient) * least_similar_sample.embed + support_gradient * query_embed
            # 更新类别原型
            self.label_to_prototype[label] = torch.mean(torch.stack([s.embed for s in sample_list]), dim=0)
        else: # 如果判断错误，那么一定要进行替换判断
            if query_minmargin > margins[min_margin_index]:
                self.demonstrations.remove(sample_list[min_margin_index])
                self.demonstrations.append(query_sample)
                self.label2sample[label].remove(sample_list[min_margin_index])
                self.label2sample[label].append(query_sample)
                # 更新类别原型
                self.label_to_prototype[label] = torch.mean(torch.stack([s.embed for s in self.label2sample[label]]), dim=0)
        assert len(self.demonstrations) == self.args.M

    def update_based_on_gradient_and_prototype(self,query_sample): # 待评估
        query_embed = query_sample.embed
        label = query_sample.label
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
        self.label_to_prototype[label] = torch.mean(torch.stack([s.embed for s in sample_list]), dim=0)
            
        assert len(self.demonstrations) == self.args.M

    def update_based_on_gradient_and_maxmargin(self,query_sample):  # 56.84
        query_embed = query_sample.embed
        label = query_sample.label
        inference_result = 1 if query_sample.pseudo_label == label else 0
        # 更新该类别的推理历史记录
        self.error_history[label].append(1 - inference_result)  # 记录错误推理

        # 计算 Support Gradient
        support_gradient = self.compute_gradient(query_sample)
        self.support_gradient_list.append(support_gradient)
        
        # 获取当前类别的样本列表
        sample_list = self.label2sample[label]

        # 计算支持集中每个样本的 margin
        margins = [self.compute_maxMargin(s.embed, label) for s in sample_list]

        # 找到 margin 最小的样本
        min_margin_index = torch.argmin(torch.tensor(margins)).item()

        least_similar_sample = sample_list[min_margin_index]
        least_similar_sample.embed = (1 - support_gradient) * least_similar_sample.embed + support_gradient * query_embed
        # 更新类别原型
        self.label_to_prototype[label] = torch.mean(torch.stack([s.embed for s in sample_list]), dim=0)
            
        assert len(self.demonstrations) == self.args.M
    
    def update_based_on_gradient_and_minmargin(self,query_sample):  # 57.34
        query_embed = query_sample.embed
        label = query_sample.label
        inference_result = 1 if query_sample.pseudo_label == label else 0
        # 更新该类别的推理历史记录
        self.error_history[label].append(1 - inference_result)  # 记录错误推理

        # 计算 Support Gradient
        support_gradient = self.compute_gradient(query_sample)
        self.support_gradient_list.append(support_gradient)
        
        # 获取当前类别的样本列表
        sample_list = self.label2sample[label]

        # 计算支持集中每个样本的 margin
        margins = [self.compute_minMargin(s.embed, label) for s in sample_list]

        # 找到 margin 最小的样本
        min_margin_index = torch.argmin(torch.tensor(margins)).item()

        least_similar_sample = sample_list[min_margin_index]
        least_similar_sample.embed = (1 - support_gradient) * least_similar_sample.embed + support_gradient * query_embed
        # 更新类别原型
        self.label_to_prototype[label] = torch.mean(torch.stack([s.embed for s in sample_list]), dim=0)
            
        assert len(self.demonstrations) == self.args.M
    
    def update_based_on_gradient_and_minmargin_topk(self,query_sample):  # 56.64
        query_embed = query_sample.embed
        label = query_sample.label
        inference_result = 1 if query_sample.pseudo_label == label else 0
        # 更新该类别的推理历史记录
        self.error_history[label].append(1 - inference_result)  # 记录错误推理

        # 计算 Support Gradient
        support_gradient = self.compute_gradient(query_sample)
        self.support_gradient_list.append(support_gradient)
        
        # 获取当前类别的样本列表
        sample_list = self.label2sample[label]

        # 计算支持集中每个样本的 margin
        margins = [self.compute_minMargin_topk(s.embed, label) for s in sample_list]

        # 找到 margin 最小的样本
        min_margin_index = torch.argmin(torch.tensor(margins)).item()

        least_similar_sample = sample_list[min_margin_index]
        least_similar_sample.embed = (1 - support_gradient) * least_similar_sample.embed + support_gradient * query_embed
        # 更新类别原型
        self.label_to_prototype[label] = torch.mean(torch.stack([s.embed for s in sample_list]), dim=0)
            
        assert len(self.demonstrations) == self.args.M
        
    def update_based_on_gradient_equal_1(self,query_sample):  # 52.5
        query_embed = query_sample.embed
        label = query_sample.label
        # 计算 Support Gradient
        support_gradient = 1
        
        # 获取当前类别的样本列表
        sample_list = self.label2sample[label]

        # 计算支持集中每个样本的 margin
        margins = [self.compute_minMargin(s.embed, label) for s in sample_list]

        # 找到 margin 最小的样本
        min_margin_index = torch.argmin(torch.tensor(margins)).item()

        least_similar_sample = sample_list[min_margin_index]
        least_similar_sample.embed = (1 - support_gradient) * least_similar_sample.embed + support_gradient * query_embed
        # 更新类别原型
        self.label_to_prototype[label] = torch.mean(torch.stack([s.embed for s in sample_list]), dim=0)
            
        assert len(self.demonstrations) == self.args.M
        
    def update_based_on_gradient_and_maxmargin_euqal_1(self,query_sample):  # 53.66
        query_embed = query_sample.embed
        label = query_sample.label
        # 计算 Support Gradient
        support_gradient = 1
        self.support_gradient_list.append(support_gradient)
        
        # 获取当前类别的样本列表
        sample_list = self.label2sample[label]

        # 计算支持集中每个样本的 margin
        margins = [self.compute_maxMargin(s.embed, label) for s in sample_list]

        # 找到 margin 最小的样本
        min_margin_index = torch.argmin(torch.tensor(margins)).item()

        least_similar_sample = sample_list[min_margin_index]
        least_similar_sample.embed = (1 - support_gradient) * least_similar_sample.embed + support_gradient * query_embed
        # 更新类别原型
        self.label_to_prototype[label] = torch.mean(torch.stack([s.embed for s in sample_list]), dim=0)
            
        assert len(self.demonstrations) == self.args.M