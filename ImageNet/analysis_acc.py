import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# 文件路径
file_path_1 = "./result/last-open_flamingo-balanced-FewShot-M=500-select_strategy=topk-update_strategy=noUpdate-.json"
file_path_2 = "./result/shuffle-last-open_flamingo-balanced-Online_ICL-M=500-select_strategy=topk-update_strategy=prototype.json"

# 读取 JSON 文件
with open(file_path_1, "r") as f:
    data_json_1 = json.load(f)

with open(file_path_2, "r") as f:
    data_json_2 = json.load(f)

# 初始化字典存储每种 label 的正确数量
label_correct_1 = defaultdict(int)
label_correct_2 = defaultdict(int)

# 统计文件 1 中每种 label 的正确数量
for data in data_json_1:
    gt_label = data["gt_label"]
    pred_label = data["pred_label"]
    
    if gt_label == pred_label:
        label_correct_1[gt_label] += 1

# 统计文件 2 中每种 label 的正确数量
for data in data_json_2:
    gt_label = data["gt_label"]
    pred_label = data["pred_label"]
    
    if gt_label == pred_label:
        label_correct_2[gt_label] += 1

# 获取所有 labels 并筛选差距超过 10 的标签
labels = []
correct_counts_1 = []
correct_counts_2 = []

for label in set(label_correct_1.keys()).union(set(label_correct_2.keys())):
    count_1 = label_correct_1[label]
    count_2 = label_correct_2[label]
    
    if abs(count_1 - count_2) >= 10:
        labels.append(label)
        correct_counts_1.append(count_1)
        correct_counts_2.append(count_2)

# 绘制柱状图
x = np.arange(len(labels))  # x 轴位置
width = 0.35  # 柱状图的宽度

fig, ax = plt.subplots(figsize=(12, 6))


# 左边的柱状图 - 文件1
bars1 = ax.bar(x - width/2, correct_counts_1, width, label='Baseline Correct')

# 右边的柱状图 - 文件2
bars2 = ax.bar(x + width/2, correct_counts_2, width, label='Prototype Correct')

# 添加标签和标题
ax.set_xlabel('Labels')
ax.set_ylabel('Correct Count')
ax.set_title('Correct Count Comparison Between baseline and prototype for Each Label')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=90)
ax.legend()

# 调整布局并保存图片
fig.tight_layout()
plt.savefig('correct_count_comparison.jpg')
#plt.show()
