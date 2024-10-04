from classification_utils import IMAGENET_CLASSNAMES_100
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# IMAGENET_CLASSNAMES_100是一个list，数据流有10000条，每100条是一类，例如0-99属于的类别是IMAGENET_CLASSNAMES_100的元素0


#file_path_1 = "./result/last-open_flamingo-balanced-FewShot-M=500-select_strategy=topk-update_strategy=noUpdate-.json"
file_path_1 = "./result/last-open_flamingo-balanced-stream=noshuffle-Online_ICL-M=500-select_strategy=topk-update_strategy=prototype.json"

# 结果文件是一个长为 5000的list，一个元素表示 数据流中2条数据被更新，那数据流中100条数据相当于 结果文件的50个元素
# 也就是说 0-49 个元素 对应的是 IMAGENET_CLASSNAMES_100【0】这一类正在被更新
# 我想统计 结果文件 每隔50对应的 IMAGENET_CLASSNAMES_100 的类别的正确数量
# 读取 JSON 文件
with open(file_path_1, "r") as f:
    data_json_1 = json.load(f)

correct  = 0
# 统计文件 1 中每种 label 的正确数量
for i,data in enumerate(data_json_1):
    gt_label = data["gt_label"]
    pred_label = data["pred_label"]
    # 计算当前数据对应的类别索引
    category_index = i // 50  # 每50个元素对应一个类别
    
    # 每50个元素输出一次准确率
    if i % 50 == 0 and i != 0:
        # 打印当前类别的正确数量
        print(f"Category: {IMAGENET_CLASSNAMES_100[category_index]}, Correct Count: {correct}")
        correct = 0  # 重置正确计数
    # 检查预测是否正确
    if gt_label == IMAGENET_CLASSNAMES_100[category_index] and gt_label == pred_label:
        correct += 1
    

# 在最后一次循环结束后输出最后一类的结果
if len(data_json_1) % 50 != 0:
    print(f"Category: {IMAGENET_CLASSNAMES_100[category_index]}, Correct Count: {correct}")