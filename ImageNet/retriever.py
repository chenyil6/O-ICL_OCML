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
        self.probs = dict()
        self.drop_prob = dict()
        self.accept_prob = dict()
        self.full_classes = set()
        self.all_meet = dict()
        self.max_samples_num = 10
        self.k = 10
    
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
            elif self.args.update_strategy == "SV_dynamic":
                self.update_based_on_SV_dynamic(samples_to_remove[0])
            elif self.args.update_strategy == "SV_weight":
                self.update_based_on_SV_weight(samples_to_remove[0])
            elif self.args.update_strategy == "SV_weight_dynamic":
                self.update_based_on_SV_weight_dynamic(samples_to_remove[0])
            elif self.args.update_strategy == "value":
                self.update_based_on_value(samples_to_remove[0],self.args.alpha)
            elif self.args.update_strategy == "value_dynamic":
                self.update_based_on_value_dynamic(samples_to_remove[0],self.args.alpha)
            else:
                print("update_strategy is not effective.")
                return
        else: # imbalanced
            if self.args.update_strategy == "CBRS":
                self.update_based_on_CBRS(samples_to_remove[0])
            if self.args.update_strategy == "CBRS_prototype":
                self.update_based_on_CBRS_prototype(samples_to_remove[0])
            elif self.args.update_strategy == "prototype":
                self.update_based_on_balance_prototype(samples_to_remove[0],self.max_samples_num)
            elif self.args.update_strategy == "SV":
                self.update_based_on_balance_SV(samples_to_remove[0],self.max_samples_num)
            elif self.args.update_strategy == "SV_dynamic":
                self.update_based_on_balance_SV_dynamic(samples_to_remove[0],self.max_samples_num)
            elif self.args.update_strategy == "SV_weight":
                self.update_based_on_balance_SV_weight(samples_to_remove[0],self.max_samples_num)
            elif self.args.update_strategy == "SV_weight_dynamic":
                self.update_based_on_balance_SV_weight_dynamic(samples_to_remove[0],self.max_samples_num)
            elif self.args.update_strategy == "value":
                self.update_based_on_balance_value(samples_to_remove[0],self.args.alpha,self.max_samples_num)
            elif self.args.update_strategy == "value_dynamic":
                self.update_based_on_balance_value_dynamic(samples_to_remove[0],self.args.alpha,self.max_samples_num)
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
    
    def update_based_on_value_dynamic(self,sample_to_remove,alpha):
        # 计算 sample_to_remove 的 value 值
        all_embeds = [s.embed for s in self.demonstrations]
        query_embed = sample_to_remove.embed.unsqueeze(0)  
        
        similarities = torch.cosine_similarity(query_embed, torch.stack(all_embeds))
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
        sample_to_remove.value = alpha * label_consistency + (1 - alpha) * proto_similarity.item()

        # 看是否要替换
        min_value_sample = min(sample_list, key=lambda s: s.value)
        if sample_to_remove.value > min_value_sample.value:
            self.replace_sample(min_value_sample,sample_to_remove,label)
            self.update_demonstrations()

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
    
    def update_based_on_SV_dynamic(self,sample_to_remove):
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
            self.update_demonstrations()

        assert len(self.demonstrations) == self.args.M

    def update_demonstrations(self,):
        if self.args.update_strategy == "SV_dynamic":
            all_embeds = [s.embed for s in self.demonstrations]
            for sample in self.demonstrations:
                query_embed = sample.embed 
                similarities = torch.cosine_similarity(query_embed.unsqueeze(0) , torch.stack(all_embeds))
                _, top_indices = torch.topk(similarities, k=11)  # 获取前11个相似样本的索引

                x1 = sum(1 for idx in top_indices[1:] if self.demonstrations[idx].label == sample.label)  
                x2 = sum(1 for idx in top_indices[1:] if self.demonstrations[idx].label != sample.label)  
                sample.value = x1-x2

        elif self.args.update_strategy == "SV_weight_dynamic":
            all_embeds = [s.embed for s in self.demonstrations]
            for sample in self.demonstrations:
                query_embed = sample.embed  
                # 计算 L2 距离
                distances = torch.norm(torch.stack(all_embeds) - query_embed, dim=1)
                weights = 1 / (distances + 1e-8)
                similarities = torch.cosine_similarity(query_embed.unsqueeze(0), torch.stack(all_embeds))
                _, top_indices = torch.topk(similarities, k=11)  

                x1 = sum(weights[idx].item() for idx in top_indices[1:] if self.demonstrations[idx].label == sample.label)  
                x2 = sum(weights[idx].item() for idx in top_indices[1:] if self.demonstrations[idx].label != sample.label)  
                sample.value = x1-x2 

        elif self.args.update_strategy == "value_dynamic":
            all_embeds = [s.embed for s in self.retriever.demonstrations]
            for sample in self.retriever.demonstrations:
                query_embed = sample.embed  
                similarities = torch.cosine_similarity(query_embed.unsqueeze(0), torch.stack(all_embeds))
                _, top_indices = torch.topk(similarities, k=11)  # 获取前11个相似样本的索引

                x1 = sum(1 for idx in top_indices[1:] if self.retriever.demonstrations[idx].label == sample.label)  
                x2 = sum(1 for idx in top_indices[1:] if self.retriever.demonstrations[idx].label != sample.label)  
                label_consistency = (x1-x2)/10.0

                # 计算对应类的prototype
                label = sample.label
                sample_list = self.label2sample[label]
                embed_list = [sample.embed for sample in sample_list]
                prototype = torch.mean(torch.stack(embed_list), dim=0)
                proto_similarity = torch.cosine_similarity(query_embed.unsqueeze(0), prototype.unsqueeze(0)).item()
                sample.value = self.args.alpha * label_consistency + (1 - self.args.alpha) * proto_similarity

    def update_based_on_SV_weight(self,sample_to_remove):
        # 计算 sample_to_remove 的 value 值
        all_embeds = [s.embed for s in self.demonstrations]
        query_embed = sample_to_remove.embed
        # 计算 L2 距离
        distances = torch.norm(torch.stack(all_embeds) - query_embed.unsqueeze(0), dim=1)
    
        # 计算权重，使用距离的倒数，避免除以零
        weights = 1 / (distances + 1e-8)  # 加一个小值防止除零错误
        similarities = torch.cosine_similarity(query_embed.unsqueeze(0), torch.stack(all_embeds))
        _, top_indices = torch.topk(similarities, k=10)  

        x1 = sum(weights[i].item() for i in top_indices if self.demonstrations[i].label == sample_to_remove.label)
        x2 = sum(weights[i].item() for i in top_indices if self.demonstrations[i].label != sample_to_remove.label)

        sample_to_remove.value = x1-x2
        
        label = sample_to_remove.label

        sample_list = self.label2sample[label]
        min_value_sample = min(sample_list, key=lambda s: s.value)
        if sample_to_remove.value > min_value_sample.value:
            self.replace_sample(min_value_sample,sample_to_remove,label)
            
        assert len(self.demonstrations) == self.args.M

    def update_based_on_SV_weight_dynamic(self,sample_to_remove):
        # 计算 sample_to_remove 的 value 值
        all_embeds = [s.embed for s in self.demonstrations]
        query_embed = sample_to_remove.embed  
        # 计算 L2 距离
        distances = torch.norm(torch.stack(all_embeds) - query_embed, dim=1)
    
        # 计算权重，使用距离的倒数，避免除以零
        weights = 1 / (distances + 1e-8)  # 加一个小值防止除零错误
        similarities = torch.cosine_similarity(query_embed.unsqueeze(0), torch.stack(all_embeds))
        _, top_indices = torch.topk(similarities, k=10)  

        x1 = sum(weights[i].item() for i in top_indices if self.demonstrations[i].label == sample_to_remove.label)
        x2 = sum(weights[i].item() for i in top_indices if self.demonstrations[i].label != sample_to_remove.label)

        sample_to_remove.value = x1-x2
        
        label = sample_to_remove.label

        sample_list = self.label2sample[label]
        min_value_sample = min(sample_list, key=lambda s: s.value)
        if sample_to_remove.value > min_value_sample.value:
            self.replace_sample(min_value_sample,sample_to_remove,label)
            self.update_demonstrations()
            
        assert len(self.demonstrations) == self.args.M

    def update_based_on_CBRS(self, sample_to_remove):
        # 原论文提出这个算法是用来解决数据流不平衡的方法
        label = sample_to_remove.label

        # 1. 更新已遇到的类别样本总数 (self.all_meet)
        if label not in self.all_meet:
            self.all_meet[label] = 0  
        self.all_meet[label] += 1  

        # 2. 判断当前类别是否为满类
        if label not in self.full_classes:
            # 如果当前类别不是满类
            # 3. 找到最大的类别（可以有多个）
            largest_count = max(self.count.values())  # 找到最大类的数据量
            largest_classes = [cls for cls, cnt in self.count.items() if cnt == largest_count]  # 所有最大类

            # 4. 在最大类中随机选择一个样本，并替换它
            # 随机选取一个最大类中的实例
            largest_class = random.choice(largest_classes)  
            sample_to_replace = random.choice(self.label2sample[largest_class])  # 随机选中一个样本

            # 5. 替换最大类中的这个样本
            self.label2sample[largest_class].remove(sample_to_replace)  # 移除旧样本
            if label not in self.label2sample:# 添加新样本
                self.label2sample[label] = [sample_to_remove] 
            else:
                self.label2sample[label].append(sample_to_remove)
            self.demonstrations.remove(sample_to_replace)  # 从总存储中移除旧样本
            self.demonstrations.append(sample_to_remove)  # 将新样本加入存储

            # 6. 更新类别样本数量
            self.count[largest_class] -= 1
            self.count[label] += 1
            
            # 7. 如果当前类别数据量和最大类相同，标记为满类
            if self.count[label] == largest_count:
                self.full_classes.add(label)  # 标记该类别为满类
            else: # 8. 如果当前类别是满类
                mc = self.count[label]  # 当前类别存储中的数量
                nc = self.all_meet[label]    # 已遇到的该类别样本总数
            
                # 9. 计算替换的概率
                probability = float(mc) / nc
                if random.uniform(0, 1) <= probability:
                    # 10. 以 mc/nc 的概率选择一个样本替换
                    sample_list = self.label2sample[label]
                    sample_to_replace = random.choice(sample_list)
                    self.label2sample[label].remove(sample_to_replace)  # 移除旧样本
                    self.label2sample[label].append(sample_to_remove)  # 添加新样本
                    self.demonstrations.remove(sample_to_replace)  # 从总存储中移除旧样本
                    self.demonstrations.append(sample_to_remove)  # 将新样本加入存储
            
            # 11. 确保内存中的样本总数等于 M
            assert len(self.demonstrations) == self.args.M
            
    def update_based_on_CBRS_prototype(self, sample_to_remove):
        # 原论文提出这个算法是用来解决数据流不平衡的方法
        label = sample_to_remove.label

        # 1. 更新已遇到的类别样本总数 (self.all_meet)
        if label not in self.all_meet:
            self.all_meet[label] = 0  
        self.all_meet[label] += 1  

        # 2. 判断当前类别是否为满类
        if label not in self.full_classes:
            # 如果当前类别不是满类
            # 3. 找到最大的类别（可以有多个）
            largest_count = max(self.count.values())  # 找到最大类的数据量
            largest_classes = [cls for cls, cnt in self.count.items() if cnt == largest_count]  # 所有最大类

            # 4. 在最大类中随机选择一个样本，并替换它
            largest_class = random.choice(largest_classes)  
            least_similar_sample,_ = self.get_least_similar_sample(self.label2sample[largest_class])

            # 5. 替换最大类中的这个样本
            self.label2sample[largest_class].remove(least_similar_sample)  
            if label not in self.label2sample:
                self.label2sample[label] = [sample_to_remove] 
            else:
                self.label2sample[label].append(sample_to_remove)
            self.demonstrations.remove(least_similar_sample)  # 从总存储中移除旧样本
            self.demonstrations.append(sample_to_remove)  # 将新样本加入存储

            # 6. 更新类别样本数量
            self.count[largest_class] -= 1
            self.count[label] += 1
            
            # 7. 如果当前类别数据量和最大类相同，标记为满类
            if self.count[label] == largest_count:
                self.full_classes.add(label)  # 标记该类别为满类
            else: # 8. 如果当前类别是满类
                mc = self.count[label]  # 当前类别存储中的数量
                nc = self.all_meet[label]    # 已遇到的该类别样本总数
            
                # 9. 计算替换的概率
                probability = float(mc) / nc
                if random.uniform(0, 1) <= probability:
                    # 10. 以 mc/nc 的概率选择一个样本替换
                    sample_list = self.label2sample[label]
                    sample_to_replace,_ = self.get_least_similar_sample(sample_list)
                    self.label2sample[label].remove(sample_to_replace)  
                    self.label2sample[label].append(sample_to_remove) 
                    self.demonstrations.remove(sample_to_replace) 
                    self.demonstrations.append(sample_to_remove)  
            
            # 11. 确保内存中的样本总数等于 M
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

    def update_based_on_balance_SV_dynamic(self,sample_to_remove,max_samples_num):
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
        self.update_demonstrations()

    def update_based_on_balance_SV_weight(self,sample_to_remove,max_samples_num):
        # 计算 sample_to_remove 的 value 值
        all_embeds = [s.embed for s in self.demonstrations]
        query_embed = sample_to_remove.embed  
        # 计算 L2 距离
        distances = torch.norm(torch.stack(all_embeds) - query_embed.unsqueeze(0), dim=1)
    
        # 计算权重，使用距离的倒数，避免除以零
        weights = 1 / (distances + 1e-8)  # 加一个小值防止除零错误
        similarities = torch.cosine_similarity(query_embed.unsqueeze(0), torch.stack(all_embeds))
        _, top_indices = torch.topk(similarities, k=self.k)  

        x1 = sum(weights[i].item() for i in top_indices if self.demonstrations[i].label == sample_to_remove.label)
        x2 = sum(weights[i].item() for i in top_indices if self.demonstrations[i].label != sample_to_remove.label)

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

    def update_based_on_balance_SV_weight_dynamic(self,sample_to_remove,max_samples_num):
        # 计算 sample_to_remove 的 value 值
        all_embeds = [s.embed for s in self.demonstrations]
        query_embed = sample_to_remove.embed  
        # 计算 L2 距离
        distances = torch.norm(torch.stack(all_embeds) - query_embed.unsqueeze(0), dim=1)
    
        # 计算权重，使用距离的倒数，避免除以零
        weights = 1 / (distances + 1e-8)  # 加一个小值防止除零错误
        similarities = torch.cosine_similarity(query_embed.unsqueeze(0), torch.stack(all_embeds))
        _, top_indices = torch.topk(similarities, k=self.k)  

        x1 = sum(weights[i].item() for i in top_indices if self.demonstrations[i].label == sample_to_remove.label)
        x2 = sum(weights[i].item() for i in top_indices if self.demonstrations[i].label != sample_to_remove.label)

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
        self.update_demonstrations()

    # 结合 value和prototype
    def update_based_on_balance_value(self,sample_to_remove,alpha,max_samples_num):
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

    def update_based_on_balance_value_dynamic(self,sample_to_remove,alpha,max_samples_num):
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
        self.update_demonstrations()

    def update_new_pjw(self,L=None,is_correct=None):
        if len(self.pool) == 0 or L is None or is_correct is None:
            # memory bank已经更新完毕了
            return
           
        # 每推理一次，剔除 self.pool 的前5个
        samples_to_remove = self.pool[:5]
        self.pool = self.pool[5:]  # 剔除前5个样本

        # 更新接受概率和丢弃概率
        if is_correct:
            self.accept_prob[L] = max(0, self.accept_prob[L] - 0.1)  # 减少接受概率
            self.drop_prob[L] = min(1, self.drop_prob[L] + 0.1)  # 增加丢弃概率
        else:
            self.accept_prob[L] = min(1, self.accept_prob[L] + 0.1)  # 增加接受概率
            self.drop_prob[L] = max(0, self.drop_prob[L] - 0.1)  # 减少丢弃概率
    
        # 遍历从 self.pool 剔除的前5个样本
        for sample in samples_to_remove:
            if len(self.demonstrations)<self.args.M:
                self.demonstrations.append(sample)
            else: # 使用 prototype 方法
                query_embed = sample.embed
                label = sample.label
                sample_list = self.label2sample[label]
                if len(sample_list) < 2:
                    # 可以直接跳过，说明 模型掌握了 这个label
                    continue
                else:  # 进行 基于 prototype 的更新
                    embed_list = [sample.embed for sample in sample_list]
                    # 计算prototype（类中心），即embed list 的平均
                    prototype = torch.mean(torch.stack(embed_list), dim=0)
                    # 计算 query_embed 和 prototype 的 cosine similarity
                    query_similarity = torch.cosine_similarity(query_embed.unsqueeze(0), prototype.unsqueeze(0)).item()
                    # 计算 embed_list 中每个元素和 prototype 的相似度
                    similarities = torch.cosine_similarity(torch.stack(embed_list), prototype.unsqueeze(0))
                    # 找到与 prototype 最不相似的元素
                    least_similar_index = torch.argmin(similarities).item()
                    least_similar_sample = sample_list[least_similar_index]
                    # 判断是否需要替换
                    if torch.rand(1).item() < self.accept_prob[label]:  # 以概率p接受新样本
                        # 遍历drop_prob这个dict，找到丢弃率最高的类
                        max_drop_label = max(self.drop_prob, key=self.drop_prob.get)
                        # 从max_drop_label类别中找到与prototype最不相似的样本进行删除
                        if len(self.label2sample[max_drop_label]) > 1:
                            embed_list_max_drop = [s.embed for s in self.label2sample[max_drop_label]]
                            prototype_max_drop = torch.mean(torch.stack(embed_list_max_drop), dim=0)
                            similarities_max_drop = torch.cosine_similarity(torch.stack(embed_list_max_drop), prototype_max_drop.unsqueeze(0))
                            least_similar_index_max_drop = torch.argmin(similarities_max_drop).item()
                            least_similar_sample_max_drop = self.label2sample[max_drop_label][least_similar_index_max_drop]
                            # 替换 memory bank 中 max_drop_label 类中最不相似的 sample
                            self.demonstrations.remove(least_similar_sample_max_drop)
                            self.label2sample[max_drop_label].remove(least_similar_sample_max_drop)
                            self.demonstrations.append(sample)
                            self.label2sample[label].append(sample)
                        else:
                            self.demonstrations.remove(least_similar_sample)
                            self.demonstrations.append(sample)      
                            self.label2sample[label].remove(least_similar_sample)
                            self.label2sample[label].append(sample)
                    elif torch.rand(1).item() < self.drop_prob[label]:  # 如果不接受，以概率q丢弃最远的样本
                        self.demonstrations.remove(least_similar_sample)
                        self.label2sample[label].remove(least_similar_sample)
    
    def update_new(self,L=None,is_correct=None,prob=0.1):
        if self.args.update_strategy == "probabilities":
            self.update_with_probabilities(L,is_correct)
        elif self.args.update_strategy == "update_old_5":
            self.update_old_5(L,is_correct,prob)
        elif self.args.update_strategy == "probabilities_new":
            self.update_Prob_and_Limit(L,is_correct,prob)
        else:
            print("update_strategy is not effective.")
            return

    def update_with_probabilities(self,L=None,is_correct=None,prob=0.1):
        if len(self.pool) == 0 or L is None or is_correct is None:
            # memory bank已经更新完毕了
            return

        if is_correct: # 如果预测对了，说明L对应的接受概率可以下降，因为表示模型对这个类别掌握了，不需要那么多的样本
            self.probs[L] = max(0, self.probs[L] - prob * (1 / (1 + self.probs[L])))
        else: # 如果预测错了，说明L对应的接受概率可以下降，因为表示模型对这个类别掌握了，不需要那么多的样本
            self.probs[L] = min(1, self.probs[L] + prob * (1 / (1 + self.probs[L]))) 
        
        samples_to_remove = self.pool[:5]
        self.pool = self.pool[5:]  # 剔除前5个样本

        # 遍历从 self.pool 剔除的前5个样本
        for sample in samples_to_remove:
            label = sample.label
            query_embed = sample.embed
            sample_list = self.label2sample[label]
            # 如果该类别样本少于2个，直接跳过
            if len(sample_list) < 2:
                continue  
            
            # Step 1: 计算当前类别的 prototype
            prototype = torch.mean(torch.stack([s.embed for s in sample_list]), dim=0)
            # 计算 query_embed 和 prototype 的 cosine similarity
            query_similarity = torch.cosine_similarity(query_embed.unsqueeze(0), prototype.unsqueeze(0)).item()

            # Step 2: 找到与 prototype 最不相似的样本
            least_similar_sample, distance = self.get_least_similar_sample(sample_list)

            # Step 3: 判断是否接受新样本，根据该类别的接受概率 (probs[label])
            if torch.rand(1).item() < self.probs[label] and distance < query_similarity:
                self.replace_sample(least_similar_sample, sample, label)
            
            # 更新当前类别的 probs 值
            if sample in self.demonstrations:
                self.probs[label] = max(0, self.probs[label] - prob * (1 / (1 + self.probs[label])))
            else:
                self.probs[label] = min(1, self.probs[label] + prob * (1 / (1 + self.probs[label]))) 

            assert len(self.demonstrations) == self.args.M
        
    def update_old_5(self,L=None,is_correct=None,prob=0.1):
        if len(self.pool) == 0 or L is None or is_correct is None:
            return

        if is_correct: # 如果预测对了，说明L对应的接受概率可以下降，因为表示模型对这个类别掌握了，不需要那么多的样本
            self.probs[L] = max(0, self.probs[L] - prob * (1 / (1 + self.probs[L])))
        else: 
            self.probs[L] = min(1, self.probs[L] + prob * (1 / (1 + self.probs[L])))

        samples_to_remove = self.pool[:5]
        self.pool = self.pool[5:]  # 剔除前5个样本

        for sample in samples_to_remove:
            if len(self.demonstrations) < self.args.M:
                self.demonstrations.append(sample)
                self.label2sample[sample.label].append(sample)
                continue
            label = sample.label
            query_embed = sample.embed
            sample_list = self.label2sample[label]
            # 如果该类别样本少于2个，直接跳过
            if len(sample_list) < 2:
                continue  
            
            prototype = torch.mean(torch.stack([s.embed for s in sample_list]), dim=0)
            query_similarity = torch.cosine_similarity(query_embed.unsqueeze(0), prototype.unsqueeze(0)).item()
            least_similar_sample, distance = self.get_least_similar_sample(sample_list)

            # Step 3: 判断是否接受新样本，根据该类别的接受概率 (probs[label])
            if torch.rand(1).item() <= self.probs[label]:
                self.demonstrations.append(sample)
                self.label2sample[sample.label].append(sample)
                self.probs[sample.label] = max(0, self.probs[sample.label] - prob * (1 / (1 + self.probs[sample.label])))
            else: # 不接受
                # 丢掉一个最远的
                self.demonstrations.remove(least_similar_sample)
                self.label2sample[sample.label].remove(least_similar_sample)
                self.probs[sample.label] = min(1, self.probs[sample.label] + prob * (1 / (1 + self.probs[sample.label])))

            while(len(self.demonstrations) > self.args.M):
                # 对 self.probs 按照value的值从小到大排序
                lower_probs_labels = sorted(self.probs, key=lambda x: self.probs[x])
                for l in lower_probs_labels:
                    if len(self.label2sample[l]) < 2:
                        continue
                    s,d = self.get_least_similar_sample(self.label2sample[l])
                    self.demonstrations.remove(s)
                    self.label2sample[l].remove(s)
                    self.probs[l] = min(1, self.probs[sample.label] + prob * (1 / (1 + self.probs[sample.label])))
                    break
    
    def update_Prob_and_Limit(self,L=None,is_correct=None,prob=0.1):
        if len(self.pool) == 0 or L is None or is_correct is None:
            return

        if is_correct: # 如果预测对了，说明L对应的接受概率可以下降，因为表示模型对这个类别掌握了，不需要那么多的样本
            self.probs[L] = max(0, self.probs[L] - prob * (1 / (1 + self.probs[L])))
        else: 
            self.probs[L] = min(1, self.probs[L] + prob * (1 / (1 + self.probs[L])))

        samples_to_remove = self.pool[:5]
        self.pool = self.pool[5:]  

        for sample in samples_to_remove:
            if len(self.demonstrations) < self.args.M:
                self.demonstrations.append(sample)
                self.label2sample[sample.label].append(sample)
                continue
            label = sample.label
            query_embed = sample.embed
            sample_list = self.label2sample[label]
            # 如果该类别样本少于2个，直接跳过
            if len(sample_list) < 3:
                continue  
            
            prototype = torch.mean(torch.stack([s.embed for s in sample_list]), dim=0)
            query_similarity = torch.cosine_similarity(query_embed.unsqueeze(0), prototype.unsqueeze(0)).item()
            least_similar_sample, distance = self.get_least_similar_sample(sample_list)

            # Step 3: 判断是否接受新样本，根据该类别的接受概率 (probs[label])
            if torch.rand(1).item() < self.probs[label]:
                if distance < query_similarity: # 接受并替换
                    self.replace_sample(least_similar_sample, sample, label)
                else: # 加入，但是要删掉 self.demonstration中 一个离它近的且label不同的样本
                    all_embed = [s.embed for s in self.demonstrations]
                    all_labels = [s.label for s in self.demonstrations]
                    similarities = torch.cosine_similarity(query_embed.unsqueeze(0), torch.stack(all_embed)).squeeze(0)
                
                    # 按相似度从高到低排序，并尝试删除标签不同的样本
                    sorted_indices = torch.argsort(similarities, descending=True)
                    for idx in sorted_indices:
                        if all_labels[idx] != label:
                            sample_to_remove = self.demonstrations[idx]
                            self.demonstrations.remove(sample_to_remove)
                            self.label2sample[sample_to_remove.label].remove(sample_to_remove)
                            break
                    self.demonstrations.append(sample)
                    self.label2sample[label].append(sample)

            else: # 不接受
                # 再根据 离prototype 的距离，来看是否要删掉最远的
                if distance < query_similarity: #(数量-1)
                    self.demonstrations.remove(least_similar_sample)
                    self.label2sample[sample.label].remove(least_similar_sample)
                    self.probs[sample.label] = min(1, self.probs[sample.label] + prob * self.probs[sample.label])

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
    
    def update_count(self,class_acc):
        if len(self.pool) == 0:
            return
        samples_to_remove = self.pool[:1]
        self.pool = self.pool[1:]
        if self.args.dataset_mode == "balanced":
            if self.args.update_strategy == "accuracy":
                self.update_based_on_accuracy(samples_to_remove[0],class_acc)
            else:
                print("update_strategy is not effective.")
                return
        else: # imbalanced
            if self.args.update_strategy == "CBRS":
                self.update_based_on_CBRS(samples_to_remove[0])
            else:
                print("update_strategy is not effective.")
                return
        
    
    def update_based_on_accuracy(self,sample,class_acc):
        label = sample.label
        accuracy = class_acc[label]
        query_embed = sample.embed
        sample_list = self.label2sample[label]
        embed_list = [sample.embed for sample in sample_list]
        prototype = torch.mean(torch.stack(embed_list), dim=0)
        query_similarity = torch.cosine_similarity(query_embed.unsqueeze(0), prototype.unsqueeze(0)).item()
        similarities = torch.cosine_similarity(torch.stack(embed_list), prototype.unsqueeze(0))
        least_similar_index = torch.argmin(similarities).item()
        sample_to_remove = sample_list[least_similar_index] # sample list中离prototype最远的样本
        if accuracy >= 0.8: # 高正确率区间：移除与类原型距离较远的样本，但至少保留min_samples个样本
            if len(sample_list) > 4:
                self.demonstrations.remove(sample_to_remove)
                self.label2sample[sample.label].remove(sample_to_remove)
            else: # 不移除 数量不变，看是否要替换
                if query_similarity > similarities[least_similar_index]:
                    self.demonstrations.remove(sample_to_remove)
                    self.demonstrations.append(sample_to_remove)
                    self.label2sample[label].remove(sample_to_remove)
                    self.label2sample[label].append(sample_to_remove)
        elif 0.5 <= accuracy < 0.8: # 中等正确率区间：替换样本
            if query_similarity > similarities[least_similar_index]:
                sample_to_remove = sample_list[least_similar_index]
                self.demonstrations.remove(sample_to_remove)
                self.demonstrations.append(sample_to_remove)
                self.label2sample[label].remove(sample_to_remove)
                self.label2sample[label].append(sample_to_remove)
        else: # 低正确率区间：增加样本
            if len(sample_list) < 20: # 如果shot小于 20 ，一定增加样本
                self.demonstrations.append(sample)
                self.label2sample[label].append(sample)
            else:
                if query_similarity > similarities[least_similar_index]:
                    sample_to_remove = sample_list[least_similar_index]
                    self.demonstrations.remove(sample_to_remove)
                    self.demonstrations.append(sample_to_remove)
                    self.label2sample[label].remove(sample_to_remove)
                    self.label2sample[label].append(sample_to_remove)
        
        while(len(self.demonstrations)>self.args.M):
            high_acc_labels = sorted(self.probs, key=lambda x: self.probs[x],reverse=True)
            for l in high_acc_labels:
                if len(self.label2sample[l]) < 4:
                    continue
                s,d = self.get_least_similar_sample(self.label2sample[l])
                self.demonstrations.remove(s)
                self.label2sample[l].remove(s)
                break