import torch
import random
from utils import get_imagenet_prompt
import os
import json
import numpy as np
import time

class DynamicReteiever:
    def __init__(self, args):
        self.args = args
        self.demonstrations = []
        self.pool = []
        self.label2sample = dict()
        self.dnum = 0
    
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
        self.dnum = min(len(self.demonstrations),4)
        if self.args.select_strategy == "random":
            indices = self.get_random(sample)
        elif self.args.select_strategy == "topk":
            indices = self.get_topk(sample)
        else:
            print("select_strategy is not effective.")
            return
        return [self.demonstrations[i] for i in indices]

    def get_random(self, sample):
        indices = random.sample(range(len(self.demonstrations)), self.dnum)
        return indices

    def get_topk(self, sample):
        demonstration_embeds = torch.stack([sample.embed for sample in self.demonstrations], dim=0)
        scores = torch.cosine_similarity(demonstration_embeds, sample.embed, dim=-1)
        values, indices = torch.topk(scores, self.dnum, largest=True)
        indices = indices.tolist()
        return indices
    
    def find_least_similar_index(self,sample_list):
        embed_list = [sample.embed for sample in sample_list]
        # 计算prototype（类中心），即embed list 的平均
        prototype = torch.mean(torch.stack(embed_list), dim=0)
        # 计算 embed_list 中每个元素和 prototype 的相似度
        similarities = torch.cosine_similarity(torch.stack(embed_list), prototype.unsqueeze(0))
        # 找到与 prototype 最不相似的元素的索引
        least_similar_index = torch.argmin(similarities).item()
        return least_similar_index
    
    def update(self):
        if len(self.pool) == 0:
            return
        samples_to_remove = self.pool[:2]
        self.pool = self.pool[2:]

        if self.args.update_strategy == "prototype":
            self.update_based_on_prototype(samples_to_remove)
        if self.args.update_strategy == "distance":
            self.update_based_on_distance(samples_to_remove)
        elif self.args.update_strategy == "global_diversity":
            self.update_based_on_global_diversity(samples_to_remove)
        elif self.args.update_strategy == "balance_random":
            self.update_based_on_balance_random(samples_to_remove)
        elif self.args.update_strategy == "balance_prototype":
            self.update_based_on_balance_prototype(samples_to_remove)
        elif self.args.update_strategy == "balance_prototype_new":
            self.update_based_on_balance_prototype_new(samples_to_remove)
        elif self.args.update_strategy == "entropy":
            self.update_based_on_entropy(samples_to_remove)
        else:
            print("update_strategy is not effective.")
            return
        
    def update_based_on_prototype(self,samples_to_remove):
        # 遍历要剔除的 2 个 sample ，根据 label 属性找到对应的 sample list
        for s in samples_to_remove:
            query_embed = s.embed
            label = s.label
            
            sample_list = self.label2sample[label]
            embed_list = [sample.embed for sample in sample_list]

            # 计算prototype（类中心），即embed list 的平均
            prototype = torch.mean(torch.stack(embed_list), dim=0)

            # 计算 query_embed 和 prototype 的 cosine similarity
            query_similarity = torch.cosine_similarity(query_embed.unsqueeze(0), prototype.unsqueeze(0)).item()

            # 计算 embed_list 中每个元素和 prototype 的相似度
            similarities = torch.cosine_similarity(torch.stack(embed_list), prototype.unsqueeze(0))

            # 找到与 prototype 最不相似的元素的索引
            least_similar_index = torch.argmin(similarities).item()

            # 判断是否需要替换
            if query_similarity > similarities[least_similar_index]:
                # 替换 memory bank 中最不相似的 sample
                self.demonstrations.remove(sample_list[least_similar_index])
                self.demonstrations.append(s)
                self.label2sample[label].remove(sample_list[least_similar_index])
                self.label2sample[label].append(s)
            
            assert len(self.demonstrations) == 500
    
    def update_based_on_distance(self,samples_to_remove):
        # 遍历要剔除的 2 个 sample ，根据 label 属性找到对应的 sample list
        for s in samples_to_remove:
            query_embed = s.embed
            label = s.label
            
            sample_list = self.label2sample[label]
            embed_list = [sample.embed for sample in sample_list]

            # 计算prototype（类中心），即embed list 的平均
            prototype = torch.mean(torch.stack(embed_list), dim=0)

            # 计算 query_embed 和 prototype 的 L2 距离
            query_distance = torch.norm(query_embed - prototype, p=2).item()

            # 计算 embed_list 中每个元素和 prototype 的 L2 距离
            distances = torch.norm(torch.stack(embed_list) - prototype, p=2, dim=1)

            # 找到与 prototype 距离最大的元素的索引
            furthest_index = torch.argmax(distances).item()
            # 判断是否需要替换
            if query_distance > distances[furthest_index]:
                # 替换 memory bank 中最不相似的 sample
                self.demonstrations.remove(sample_list[furthest_index])
                self.demonstrations.append(s)
                self.label2sample[label].remove(sample_list[furthest_index])
                self.label2sample[label].append(s)
            
            assert len(self.demonstrations) == 500

    def update_based_on_global_diversity(self, samples_to_remove):
        """基于全局多样性的更新策略"""

        for sample in samples_to_remove:
            label = sample.label
        
            # 获取该类别的样本列表
            sample_list = self.label2sample[label]
            embed_list = [s.embed for s in sample_list]

            # 计算类别原型（prototype）
            prototype = torch.mean(torch.stack(embed_list), dim=0)

            # 计算新样本与该类别中所有样本的相似度
            query_similarity = torch.cosine_similarity(sample.embed.unsqueeze(0), torch.stack(embed_list), dim=-1)
        
            # 计算新样本与其他类别的原型的相似度
            other_prototypes = [torch.mean(torch.stack([s.embed for s in self.label2sample[other_label]]), dim=0)
                            for other_label in self.label2sample if other_label != label]
            query_other_similarity = torch.cosine_similarity(sample.embed.unsqueeze(0), torch.stack(other_prototypes), dim=-1)
            #print("torch.max(query_other_similarity):",torch.max(query_other_similarity))
            # 如果新样本与其他类别的原型相似度太高，直接丢弃（保持类别间分离性）
            #if torch.max(query_other_similarity) > self.args.similarity_threshold:
            if torch.max(query_other_similarity) > 0.7:
                continue

            # 如果新样本与该类别中样本的相似度较低，考虑加入 Memory Bank
            #if torch.min(query_similarity) < self.args.diversity_threshold:
            #print("torch.min(query_similarity):",torch.min(query_similarity))
            if torch.min(query_similarity) < 0.3:
                # 找到与 prototype 最相似的样本的索引
                similarities = torch.cosine_similarity(torch.stack(embed_list), prototype.unsqueeze(0))
                least_similar_index = torch.argmin(similarities).item()

                # 替换最相似的样本，保持类内多样性
                self.demonstrations.remove(sample_list[least_similar_index])
                self.demonstrations.append(sample)
                self.label2sample[label].remove(sample_list[least_similar_index])
                self.label2sample[label].append(sample)
            
            # 确保 memory bank 大小保持一致
            assert len(self.demonstrations) == 500

    def update_based_on_balance_random(self,samples_to_remove):
        # 遍历 samples_to_remove
        for sample in samples_to_remove:
            label = sample.label

            # 如果这个类不在 self.label2sample 中, 则直接加入到memory bank中，
            if label not in self.label2sample:
                self.demonstrations.append(sample)
                self.label2sample[label] = [sample]
                for key,samples_list in self.label2sample.items():
                    if len(samples_list) > 5:
                        # 随机删掉一个
                        removed_sample = random.choice(samples_list)
                        self.label2sample[key].remove(removed_sample)
                        self.demonstrations.remove(removed_sample)
                        break
            elif len(self.label2sample[label]) < 5:
                self.demonstrations.append(sample)
                self.label2sample[label].append(sample)
                for key,samples_list in self.label2sample.items():
                    if len(samples_list) > 5:
                        # 随机删掉一个
                        removed_sample = random.choice(samples_list)
                        self.label2sample[key].remove(removed_sample)
                        self.demonstrations.remove(removed_sample)
                        break
            else: #随机换掉一个
                sample_list = self.label2sample[label]
                removed_sample = random.choice(sample_list)
                self.label2sample[label].remove(removed_sample)
                self.demonstrations.remove(removed_sample)
                self.demonstrations.append(sample)
                self.label2sample[label].append(sample)

            assert len(self.demonstrations) == 500

    def update_based_on_balance_prototype(self,samples_to_remove):
        # 遍历 samples_to_remove
        for sample in samples_to_remove:
            label = sample.label
            query_embed = sample.embed
            # 如果这个类不在 self.label2sample 中, 则直接加入到memory bank中，
            if label not in self.label2sample:
                self.demonstrations.append(sample)
                self.label2sample[label] = [sample]
                for key,sample_list in self.label2sample.items():
                    if len(sample_list) > 5:
                        # 删掉离中心点（prototype）最远的
                        least_similar_index = self.find_least_similar_index(sample_list)
                        removed_sample = sample_list[least_similar_index]
                        self.label2sample[key].remove(removed_sample)
                        self.demonstrations.remove(removed_sample)
                        break
            elif len(self.label2sample[label]) < 5:
                self.demonstrations.append(sample)
                self.label2sample[label].append(sample)
                for key,sample_list in self.label2sample.items():
                    if len(sample_list) > 5:
                        # 删掉离中心点（prototype）最远的
                        least_similar_index = self.find_least_similar_index(sample_list)
                        removed_sample = sample_list[least_similar_index]
                        self.label2sample[key].remove(removed_sample)
                        self.demonstrations.remove(removed_sample)
                        break
            else: # 删掉离中心点（prototype）最远的
                sample_list = self.label2sample[label]
                embed_list = [sample.embed for sample in sample_list]

                # 计算prototype（类中心），即embed list 的平均
                prototype = torch.mean(torch.stack(embed_list), dim=0)

                # 计算 query_embed 和 prototype 的 cosine similarity
                query_similarity = torch.cosine_similarity(query_embed.unsqueeze(0), prototype.unsqueeze(0)).item()

                # 计算 embed_list 中每个元素和 prototype 的相似度
                similarities = torch.cosine_similarity(torch.stack(embed_list), prototype.unsqueeze(0))

                # 找到与 prototype 最不相似的元素的索引
                least_similar_index = torch.argmin(similarities).item()

                # 判断是否需要替换
                if query_similarity > similarities[least_similar_index]:
                    # 替换 memory bank 中最不相似的 sample
                    self.demonstrations.remove(sample_list[least_similar_index])
                    self.demonstrations.append(sample)
                    self.label2sample[label].remove(sample_list[least_similar_index])
                    self.label2sample[label].append(sample)

    def update_based_on_balance_prototype_new(self, samples_to_remove):
        for sample in samples_to_remove:
            label = sample.label
            query_embed = sample.embed

            # 类别不在 memory bank 中，直接加入
            if label not in self.label2sample:
                self.demonstrations.append(sample)
                self.label2sample[label] = [sample]
        
            # 类别样本数小于5，直接加入
            elif len(self.label2sample[label]) < 5:
                self.demonstrations.append(sample)
                self.label2sample[label].append(sample)
        
            else:
                # 找到类内与prototype最不相似的样本
                sample_list = self.label2sample[label]
                embed_list = [s.embed for s in sample_list]

                # 计算 prototype（类中心）
                prototype = torch.mean(torch.stack(embed_list), dim=0)

                # 计算 query_embed 和 prototype 的相似度
                query_similarity = torch.cosine_similarity(query_embed.unsqueeze(0), prototype.unsqueeze(0)).item()

                # 计算类内每个样本与 prototype 的相似度
                similarities = torch.cosine_similarity(torch.stack(embed_list), prototype.unsqueeze(0))

                # 找到与 prototype 最不相似的样本
                least_similar_index = torch.argmin(similarities).item()

                # 如果新样本与 prototype 更相似，则替换最不相似的样本
                if query_similarity > similarities[least_similar_index]:
                    removed_sample = sample_list[least_similar_index]
                    self.label2sample[label].remove(removed_sample)
                    self.demonstrations.remove(removed_sample)
                    self.demonstrations.append(sample)
                    self.label2sample[label].append(sample)

            # 保证 memory bank 总数不超过 500
            while len(self.demonstrations) > 500:
                # 从最大类别随机删除一个样本
                max_label = max(self.label2sample, key=lambda k: len(self.label2sample[k]))
                removed_sample = random.choice(self.label2sample[max_label])
                self.label2sample[max_label].remove(removed_sample)
                self.demonstrations.remove(removed_sample)

    def update_based_on_entropy(self, samples_to_remove):
        for sample in samples_to_remove:
            label = sample.label
        
            # 类别不存在时，直接添加
            if label not in self.label2sample:
                self.demonstrations.append(sample)
                self.label2sample[label] = [sample]
            else:
                sample_list = self.label2sample[label]
            
                # 计算类内熵
                entropy = self.compute_entropy(sample_list)
            
                # 如果类内样本数小于5，直接加入
                if len(sample_list) < 5:
                    self.demonstrations.append(sample)
                    self.label2sample[label].append(sample)
                else:
                    # 如果新样本能提升熵值，则替换类内最不具代表性的样本
                    new_entropy = self.compute_entropy(sample_list + [sample])
                    if new_entropy > entropy:
                        least_representative = self.find_least_similar_index(sample_list)
                        self.demonstrations.remove(sample_list[least_representative])
                        self.label2sample[label].remove(sample_list[least_representative])
                        self.label2sample[label].append(sample)
                        self.demonstrations.append(sample)

    def compute_entropy(self, sample_list):
        # 根据相似度计算样本的熵
        embed_list = [s.embed for s in sample_list]
        distances = torch.pdist(torch.stack(embed_list))
        return -torch.sum(distances * torch.log(distances + 1e-5))  # 熵的简单估计

