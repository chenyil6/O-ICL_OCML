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
            elif self.args.update_strategy == "clip":
                self.update_based_on_clip(samples_to_remove[0])
            elif self.args.update_strategy == "combined":
                self.update_based_on_combined(samples_to_remove[0])
            elif self.args.update_strategy == "margin":
                self.update_based_on_margin(samples_to_remove[0])
            else:
                print(f"{self.args.update_strategy} is not effective.")
                return
        else: # imbalanced
            if self.args.update_strategy == "prototype":
                self.update_based_on_balance_prototype(samples_to_remove[0],self.max_samples_num)
            else:
                print("update_strategy is not effective.")
                return
        
    def update_based_on_prototype(self,sample_to_remove):
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
    
    def update_based_on_clip(self,sample_to_remove):
        query_similarity = sample_to_remove.similarity
        label = sample_to_remove.label

        sample_list = self.label2sample[label]
        similarity_list = torch.tensor([sample.similarity for sample in sample_list])  

        least_similar_index = torch.argmin(similarity_list).item()

        # 判断是否需要替换
        if query_similarity > similarity_list[least_similar_index]:
            self.demonstrations.remove(sample_list[least_similar_index])
            self.demonstrations.append(sample_to_remove)
            self.label2sample[label].remove(sample_list[least_similar_index])
            self.label2sample[label].append(sample_to_remove)
            
        assert len(self.demonstrations) == self.args.M
    
    def update_based_on_combined(self,sample_to_remove,alpha=0.7):
        """
        结合 Prototype 相似度和 Clip 相似度进行样本更新 alpha 控制两者权重。
        """
        query_embed = sample_to_remove.embed
        query_similarity_clip = sample_to_remove.similarity  
        label = sample_to_remove.label

        # 获取当前类别的样本列表和嵌入向量
        sample_list = self.label2sample[label]
        embed_list = [sample.embed for sample in sample_list]
        prototype = torch.mean(torch.stack(embed_list), dim=0)
        query_similarity_prototype = torch.cosine_similarity(query_embed.unsqueeze(0), prototype.unsqueeze(0)).item()
        combined_query_similarity = alpha * query_similarity_prototype + (1 - alpha) * query_similarity_clip
        similarities_prototype = torch.cosine_similarity(torch.stack(embed_list), prototype.unsqueeze(0))
        
        # 对于每个类内样本，计算综合相似度
        combined_similarities = []
        for i, sample in enumerate(sample_list):
            # 每个样本的 Clip 相似度
            similarity_clip = sample.similarity
            # 每个样本的 Prototype 相似度
            similarity_prototype = similarities_prototype[i].item()
            # 计算综合相似度
            combined_similarity = alpha * similarity_prototype + (1 - alpha) * similarity_clip
            combined_similarities.append(combined_similarity)

        # 找到类内综合相似度最低的样本
        combined_similarities = torch.tensor(combined_similarities)
        least_similar_index = torch.argmin(combined_similarities).item()

        # 判断是否需要替换
        if combined_query_similarity > combined_similarities[least_similar_index]:
            self.demonstrations.remove(sample_list[least_similar_index])
            self.demonstrations.append(sample_to_remove)
            self.label2sample[label].remove(sample_list[least_similar_index])
            self.label2sample[label].append(sample_to_remove)
        
        assert len(self.demonstrations) == self.args.M
    
    def update_based_on_margin(self, sample_to_remove):
        query_embed = sample_to_remove.embed
        label = sample_to_remove.label
        sample_list = self.label2sample[label]
        embed_list = [sample.embed for sample in sample_list]
        current_prototype = self.label_to_prototype[label]
        similarities_prototype = torch.cosine_similarity(torch.stack(embed_list), current_prototype.unsqueeze(0))
        query_similarity = torch.cosine_similarity(query_embed.unsqueeze(0), current_prototype.unsqueeze(0)).item()

        query_other_similarities = [
            torch.cosine_similarity(query_embed.unsqueeze(0), proto.unsqueeze(0)).item()
            for lbl, proto in self.label_to_prototype.items() if lbl != label
        ]
        query_min_other_similarity = min(query_other_similarities) if other_similarities else 0

        combined_query_similarity = query_similarity - query_min_other_similarity

        # 接着找到所属类，计算所属类样本 离prototype的相似度减去 min_other_similarity
        combined_similarities = []
        for i, sample in enumerate(sample_list):
            similarity_prototype = similarities_prototype[i].item()
            other_similarities = [
            torch.cosine_similarity(sample.embed.unsqueeze(0), proto.unsqueeze(0)).item()
                for lbl, proto in self.label_to_prototype.items() if lbl != label
            ]
            min_other_similarity = min(other_similarities) if other_similarities else 0

            combined_similarity = similarity_prototype - min_other_similarity
            combined_similarities.append(combined_similarity)
        
        combined_similarities = torch.tensor(combined_similarities)
        least_similar_index = torch.argmin(combined_similarities).item()

        if combined_query_similarity > combined_similarities[least_similar_index]:
            self.demonstrations.remove(sample_list[least_similar_index])
            self.demonstrations.append(sample_to_remove)
            self.label2sample[label].remove(sample_list[least_similar_index])
            self.label2sample[label].append(sample_to_remove)

            new_sample_list = self.label2sample[label]
            new_embeddings = torch.stack([s.embed for s in new_sample_list])
            self.label_to_prototype[label] = torch.mean(new_embeddings, dim=0)
        
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
        if self.args.update_strategy == "default":
            self.update_based_on_default(query_sample)
        elif self.args.update_strategy == "gradient_maxmargin":
            self.update_based_on_gradient_and_maxmargin(query_sample)
        elif self.args.update_strategy == "gradient_minmargin":
            self.update_based_on_gradient_and_minmargin(query_sample)
        elif self.args.update_strategy == "gradient_equal_1":
            self.update_based_on_gradient_equal_1(query_sample)
        elif self.args.update_strategy == "rate":
            self.update_based_on_rate(query_sample)
        elif self.args.update_strategy == "inference":
            self.update_based_on_inference(query_sample)
        elif self.args.update_strategy == "maxMargin":
            self.update_based_on_max_margin(query_sample)
        elif self.args.update_strategy == "minMargin":
            self.update_based_on_min_margin(query_sample)
        else:
            print("update_strategy is not effective.")
            return
        
    def compute_support_gradient(self, confidence, inference_result, label):
        """
        计算 Support Gradient 用于更新支持集
        confidence: 当前推理的置信度
        inference_result: 当前推理的结果 (1 表示正确 0 表示错误)
        label: 当前推理样本的类别
        """
        alpha = 0.8  # 置信度的影响较大
        gamma = 0.001 # 基础更新项较小
        delta = 0.2  # 历史错误率的影响中等

        # 获取该类别的历史错误率
        error_rate = sum(self.error_history[label]) / len(self.error_history[label]) if len(self.error_history[label]) > 0 else 0

        # 基础 Support Gradient 公式
        support_gradient = alpha * (1 - confidence) + gamma

        # 考虑历史错误推理的影响
        support_gradient += delta * error_rate

        return support_gradient

    def compute_rate(self,sample):
        alpha = 0.4
        beta = 0.2
        delta = 0.4

        confidence = sample.gt_score
        clip_similairity = (sample.similarity+1)/2
        margin = self.compute_margin(sample.embed,sample.label)
        margin = (margin+2)/4
        rate = alpha * (1 - confidence) + beta *clip_similairity  + delta * margin

        return rate
    
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

        # 类间相似度 (与其他类别的原型的最小相似度)
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
    
    def update_based_on_default(self,query_sample):  # self.compute_support_gradient： 46.32
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
    
    def update_based_on_gradient_and_minmargin(self,query_sample):  
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
        
    def update_based_on_gradient_equal_1(self,query_sample):  
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
        

    def update_based_on_inference(self,query_sample): 
        query_embed = query_sample.embed
        label = query_sample.label
        query_clip_similarity = query_sample.similarity

        sample_list = self.label2sample[label]
        embed_list = [sample.embed for sample in sample_list]
        clip_similarity_list = torch.tensor([sample.similarity for sample in sample_list]+ [query_clip_similarity])  

        prototype = torch.mean(torch.stack(embed_list), dim=0)

        query_prototype_similarity = torch.cosine_similarity(query_embed.unsqueeze(0), prototype.unsqueeze(0)).item()

        prototype_similarities = torch.cosine_similarity(torch.stack(embed_list), prototype.unsqueeze(0))

        least_similar_index = torch.argmin(prototype_similarities).item()

        # 如果推理结果正确，增加相似度
        if query_sample.pseudo_label == label:
            # 根据置信度来做判断替换
            if torch.rand(1).item() < query_sample.gt_score:
                if query_prototype_similarity > prototype_similarities[least_similar_index]:
                    self.demonstrations.remove(sample_list[least_similar_index])
                    self.demonstrations.append(query_sample)
                    self.label2sample[label].remove(sample_list[least_similar_index])
                    self.label2sample[label].append(query_sample)
        # 如果推理结果错误
        else:
            # 根据样本相对 sample_list 中是否是困难样本来做替换判断
            mean_similarity = clip_similarity_list.mean().item()
            std_similarity = clip_similarity_list.std().item()
            # 判断 query 样本是否为困难样本：根据批次均值和标准差判断
            normalized_similarity = (query_clip_similarity - mean_similarity) / (std_similarity + 1e-8)  # 防止除以0
            alpha = -0.842  # 标准正态分布的20%分位数

            # 如果不是困难样本，则替换掉 sample list中离 prototype 最远的样本
            if normalized_similarity > alpha:
                if query_prototype_similarity > prototype_similarities[least_similar_index]:
                    self.demonstrations.remove(sample_list[least_similar_index])
                    self.demonstrations.append(query_sample)
                    self.label2sample[label].remove(sample_list[least_similar_index])
                    self.label2sample[label].append(query_sample)

        assert len(self.demonstrations) == self.args.M
    
    def update_based_on_rate(self,query_sample):
        label = query_sample.label

        rate = self.compute_rate(query_sample)
        # 获取当前类别的样本列表
        sample_list = self.label2sample[label]

        # 计算支持集中每个样本的 margin
        margins = [self.compute_margin(s.embed, label) for s in sample_list]

        # 找到 margin 最小的样本
        min_margin_index = torch.argmin(torch.tensor(margins)).item()
        if torch.rand(1).item() < rate:
            self.demonstrations.remove(sample_list[min_margin_index])
            self.demonstrations.append(query_sample)
            sample_list[min_margin_index] = query_sample

            # 更新类别原型
            self.label_to_prototype[label] = torch.mean(torch.stack([s.embed for s in sample_list]), dim=0)

    def update_based_on_min_margin(self,query_sample):
        update_threshold = torch.quantile(torch.tensor(self.support_gradient_list), 0.8).item()  
        query_embed = query_sample.embed
        label = query_sample.label
        inference_result = 1 if query_sample.pseudo_label == label else 0
        confidence = query_sample.gt_score
        # 更新该类别的推理历史记录
        self.error_history[label].append(1 - inference_result)  # 记录错误推理
        # 计算 Support Gradient
        support_gradient = self.compute_support_gradient(confidence, inference_result, label)
        self.support_gradient_list.append(support_gradient)
        if support_gradient > update_threshold:
            # 计算当前数据流样本的 margin
            stream_margin = self.compute_margin(query_embed, label)

            # 获取当前类别的样本列表
            sample_list = self.label2sample[label]

            # 计算支持集中每个样本的 margin
            margins = [self.compute_margin(s.embed, label) for s in sample_list]

            # 找到 margin 最小的样本
            min_margin_index = torch.argmin(torch.tensor(margins)).item()

            # 如果数据流样本的 margin 更大，则替换支持集中的样本
            if stream_margin > margins[min_margin_index]:
                # 替换支持集中的样本
                self.demonstrations.remove(sample_list[min_margin_index])
                self.demonstrations.append(query_sample)
                sample_list[min_margin_index] = query_sample

                # 更新类别原型
                self.label_to_prototype[label] = torch.mean(torch.stack([s.embed for s in sample_list]), dim=0)

    def update_based_on_max_margin(self, query_sample): 
        query_embed = query_sample.embed
        label = query_sample.label

        current_prototype = self.label_to_prototype[label]

        query_similarity = torch.cosine_similarity(query_embed.unsqueeze(0), current_prototype.unsqueeze(0)).item()

        other_similarities = [
            torch.cosine_similarity(query_embed.unsqueeze(0), proto.unsqueeze(0)).item()
            for lbl, proto in self.label_to_prototype.items() if lbl != label
        ]
        min_other_similarity = min(other_similarities) if other_similarities else 0

        if query_sample.pseudo_label == label:
            query_similarity += query_sample.gt_score
        else:
            query_similarity += query_sample.gt_score*0.5

        # 找到当前类别中与原型最不相似的样本
        sample_list = self.label2sample[label]
        similarities = torch.cosine_similarity(torch.stack([s.embed for s in sample_list]), current_prototype.unsqueeze(0))
        least_similar_index = torch.argmin(similarities).item()

        # 判断是否需要替换
        if query_similarity - min_other_similarity > similarities[least_similar_index]:
            # 移除最不相似的样本
            old_sample = sample_list[least_similar_index]
            self.label2sample[label].remove(old_sample)
            self.demonstrations.remove(old_sample)

            # 添加新样本
            self.demonstrations.append(query_sample)
            sample_list.append(query_sample)

            # 更新原型
            new_embeddings = torch.stack([s.embed for s in sample_list])
            self.label_to_prototype[label] = torch.mean(new_embeddings, dim=0)

        assert len(self.demonstrations) == self.args.M
    
    def update_online_based_on_margin(self, query_sample): 
        query_embed = query_sample.embed
        label = query_sample.label

        stream_mrigin = self.compute_margin(query_embed,label)
        
        if query_sample.pseudo_label == label:
            stream_mrigin += query_sample.gt_score
        else:
            stream_mrigin += query_sample.gt_score*0.5

        # 找到当前类别中与原型最不相似的样本
        sample_list = self.label2sample[label]
        margins = [self.compute_margin(s.embed, label) for s in sample_list]

        # 找到 margin 最小的样本
        min_margin_index = torch.argmin(torch.tensor(margins)).item()
        # 判断是否需要替换
        if stream_mrigin > margins[min_margin_index]:
            # 移除最不相似的样本
            old_sample = sample_list[min_margin_index]
            self.label2sample[label].remove(old_sample)
            self.demonstrations.remove(old_sample)

            # 添加新样本
            self.demonstrations.append(query_sample)
            sample_list.append(query_sample)

            # 更新原型
            new_embeddings = torch.stack([s.embed for s in sample_list])
            self.label_to_prototype[label] = torch.mean(new_embeddings, dim=0)

        assert len(self.demonstrations) == self.args.M