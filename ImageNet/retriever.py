import torch
import random
from utils import get_imagenet_prompt
import os
import json
import numpy as np

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
        elif self.args.update_strategy == "balance_random":
            self.update_based_on_balance_random(samples_to_remove)
        elif self.args.update_strategy == "balance_prototype":
            self.update_based_on_balance_prototype(samples_to_remove)
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