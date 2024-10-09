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
        self.dnum = 4
        self.label_to_prototype = {}
    
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

    def get_topk_l2(self, sample):
        # 获取所有 demonstration 的 embed
        demonstration_embeds = torch.stack([sample.embed for sample in self.demonstrations], dim=0)
        # 计算每个 demonstration 和当前 sample 的 L2 distance
        distances = torch.norm(demonstration_embeds - sample.embed.unsqueeze(0), p=2, dim=1)
        # 取最小的 k 个距离对应的索引
        values, indices = torch.topk(distances, self.dnum, largest=False)
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
            elif self.args.update_strategy == "SV":
                self.update_based_on_SV(samples_to_remove[0])
            elif self.args.update_strategy == "value":
                self.update_based_on_value(samples_to_remove[0],self.args.alpha)
            elif self.args.update_strategy == "clip":
                self.update_based_on_embed(samples_to_remove[0])
            else:
                print("update_strategy is not effective.")
                return
        else: # imbalanced
            if self.args.update_strategy == "prototype":
                self.update_based_on_balance_prototype(samples_to_remove[0],self.max_samples_num)
            elif self.args.update_strategy == "SV":
                self.update_based_on_balance_SV(samples_to_remove[0],self.max_samples_num)
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
    
    def update_based_on_embed(self,sample_to_remove):
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
    # 结合 value和prototype
    def update_based_on_value(self,sample_to_remove,alpha):
        # 计算 sample_to_remove 的 value 值
        all_embeds = [s.embed for s in self.demonstrations]
        query_embed = sample_to_remove.embed  
        
        similarities = torch.cosine_similarity(query_embed.unsqueeze(0), torch.stack(all_embeds))
        _, top_indices = torch.topk(similarities, k=10)  

        x1 = sum(1 for i in top_indices if self.demonstrations[i].label == sample_to_remove.label)
        x2 = sum(1 for i in top_indices if self.demonstrations[i].label != sample_to_remove.label)

        label_consistency = (x1 - x2) / 10.0 
        
        # 计算对应类的prototype
        label = sample_to_remove.label
        sample_list = self.label2sample[label]
        embed_list = [sample.embed for sample in sample_list]
        prototype = torch.mean(torch.stack(embed_list), dim=0)
        proto_similarity = torch.cosine_similarity(query_embed.unsqueeze(0), prototype.unsqueeze(0)).item()
        sample_to_remove.value = alpha * label_consistency + (1 - alpha) * proto_similarity

        # 看是否要替换
        min_value_sample = min(sample_list, key=lambda s: s.value)
        if sample_to_remove.value > min_value_sample.value:
            self.replace_sample(min_value_sample,sample_to_remove,label)

        assert len(self.demonstrations) == self.args.M
    
    def update_based_on_SV(self,sample_to_remove):
        # 计算 sample_to_remove 的 value 值
        all_embeds = [s.embed for s in self.demonstrations]
        query_embed = sample_to_remove.embed
        
        similarities = torch.cosine_similarity(query_embed.unsqueeze(0), torch.stack(all_embeds))
        _, top_indices = torch.topk(similarities, k=10)  

        x1 = sum(1 for i in top_indices if self.demonstrations[i].label == sample_to_remove.label)
        x2 = sum(1 for i in top_indices if self.demonstrations[i].label != sample_to_remove.label)

        sample_to_remove.value = x1-x2
        
        label = sample_to_remove.label
    
        sample_list = self.label2sample[label]
        min_value_sample = min(sample_list, key=lambda s: s.value)
        if sample_to_remove.value > min_value_sample.value:
            self.replace_sample(min_value_sample,sample_to_remove,label)

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

    def update_based_on_balance_SV(self,sample_to_remove,max_samples_num):
        all_embeds = [s.embed for s in self.demonstrations]
        query_embed = sample_to_remove.embed  
        
        similarities = torch.cosine_similarity(query_embed.unsqueeze(0), torch.stack(all_embeds))
        _, top_indices = torch.topk(similarities, k=self.k)  

        x1 = sum(1 for i in top_indices if self.demonstrations[i].label == sample_to_remove.label)
        x2 = sum(1 for i in top_indices if self.demonstrations[i].label != sample_to_remove.label)

        sample_to_remove.value = x1-x2
        
        label = sample_to_remove.label
        if label not in self.label2sample: #直接加入
            self.label2sample[label] = [sample_to_remove]
            self.demonstrations.append(sample_to_remove)
        elif len(self.label2sample[label]) < max_samples_num: #直接加入
            self.label2sample[label].append(sample_to_remove)
            self.demonstrations.append(sample_to_remove)
        else:
            sample_list = self.label2sample[label]
            min_value_sample = min(sample_list, key=lambda s: s.value)
            if sample_to_remove.value > min_value_sample.value:
                self.replace_sample(min_value_sample,sample_to_remove,label)

        while(len(self.demonstrations) > self.args.M):
            # 从最大类别中删掉一个value值最低的样本
            max_label = max(self.label2sample, key=lambda k: len(self.label2sample[k]))
            max_sample_list = self.label2sample[max_label]
            removed_sample = min(max_sample_list, key=lambda sample: sample.value)
            self.label2sample[max_label].remove(removed_sample)
            self.demonstrations.remove(removed_sample)
                
        assert len(self.demonstrations) == self.args.M

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
        elif self.args.update_strategy == "inference":
            self.update_based_on_inference(query_sample)
        elif self.args.update_strategy == "margin":
            self.update_based_on_max_margin(query_sample)
        else:
            print("update_strategy is not effective.")
            return
    
    def update_based_on_default(self,query_sample): 
        query_embed = query_sample.embed
        label = query_sample.label

        sample_list = self.label2sample[label]
        embed_list = [sample.embed for sample in sample_list]

        prototype = torch.mean(torch.stack(embed_list), dim=0)

        query_similarity = torch.cosine_similarity(query_embed.unsqueeze(0), prototype.unsqueeze(0)).item()

        similarities = torch.cosine_similarity(torch.stack(embed_list), prototype.unsqueeze(0))

        least_similar_index = torch.argmin(similarities).item()

        if query_sample.pseudo_label == label:
            # 有0.8的概率作替换判断
            if torch.rand(1).item() < 0.8:
                if query_similarity > similarities[least_similar_index]:
                    self.demonstrations.remove(sample_list[least_similar_index])
                    self.demonstrations.append(query_sample)
                    self.label2sample[label].remove(sample_list[least_similar_index])
                    self.label2sample[label].append(query_sample)
        else: # 如果判断错误，那么一定要进行替换判断
            if query_similarity > similarities[least_similar_index]:
                self.demonstrations.remove(sample_list[least_similar_index])
                self.demonstrations.append(query_sample)
                self.label2sample[label].remove(sample_list[least_similar_index])
                self.label2sample[label].append(query_sample)
        assert len(self.demonstrations) == self.args.M
        
    def update_based_on_inference(self,query_sample): # 这个策略不ok （inference）
        query_embed = query_sample.embed
        label = query_sample.label

        sample_list = self.label2sample[label]
        embed_list = [sample.embed for sample in sample_list]

        prototype = torch.mean(torch.stack(embed_list), dim=0)

        query_similarity = torch.cosine_similarity(query_embed.unsqueeze(0), prototype.unsqueeze(0)).item()

        similarities = torch.cosine_similarity(torch.stack(embed_list), prototype.unsqueeze(0))

        least_similar_index = torch.argmin(similarities).item()

        # 如果推理结果正确，增加相似度
        if query_sample.pseudo_label == label:
            query_similarity += query_sample.pred_score[0]
        # 如果推理结果错误，也增加相似度，但可能使用一个不同的权重
        else:
            query_similarity += query_sample.pred_score[0] * 0.5

        # 判断是否需要替换
        if query_similarity > similarities[least_similar_index]:
            self.demonstrations.remove(sample_list[least_similar_index])
            self.demonstrations.append(query_sample)
            self.label2sample[label].remove(sample_list[least_similar_index])
            self.label2sample[label].append(query_sample)

        assert len(self.demonstrations) == self.args.M
        
    def update_based_on_max_margin(self, query_sample): 
        query_embed = query_sample.embed
        label = query_sample.label

        current_prototype = self.label_to_prototype[label]

        query_similarity = torch.cosine_similarity(query_embed.unsqueeze(0), current_prototype.unsqueeze(0)).item()

        # 计算新样本与其他类别的原型的相似度
        other_similarities = [
            torch.cosine_similarity(query_embed.unsqueeze(0), proto.unsqueeze(0)).item()
            for lbl, proto in self.label_to_prototype.items() if lbl != label
        ]
        min_other_similarity = min(other_similarities) if other_similarities else 0

        # 如果推理结果正确，增加相似度
        if query_sample.pseudo_label == label:
            query_similarity += query_sample.gt_score[0]
        # 如果推理结果错误，也增加相似度，但可能使用一个不同的权重
        else:
            query_similarity += query_sample.gt_score[0] * 0.5

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

    def add_sample_to_support_set(self, sample):
        label = sample.label
        self.demonstrations.append(sample)
        self.label2sample[label].append(sample)
    
        # 增加样本后更新原型（增量计算）
        current_prototype = self.label_to_prototype.get(label, torch.zeros_like(sample.embed))
        new_prototype = (current_prototype * (len(self.label2sample[label]) - 1) + sample.embed) / len(self.label2sample[label])
        self.label_to_prototype[label] = new_prototype

    def remove_sample_from_support_set(self, sample):
        label = sample.label
        self.demonstrations.remove(sample)
        self.label2sample[label].remove(sample)
    
        if self.label2sample[label]:
            # 重新计算原型
            new_embeddings = torch.stack([s.embed for s in self.label2sample[label]])
            new_prototype = torch.mean(new_embeddings, dim=0)
            self.label_to_prototype[label] = new_prototype
        else:
            # 处理没有样本的类别（可选）
            self.label_to_prototype[label] = torch.zeros_like(next(iter(self.label_to_prototype.values())))
