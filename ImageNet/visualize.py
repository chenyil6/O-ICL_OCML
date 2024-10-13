import json
import matplotlib.pyplot as plt
path = './result/open_flamingo-balanced-Online_ICL_Old-M=1000-select_strategy=cosine-update_strategy=clip.json'

with open(path, "r") as f:
    data = json.load(f)

# 提取所有预测结果的置信度分数
gt_scores = [d["gt_score"] for d in data if d["gt_label"] == d["pred_label"]]

# 可视化置信度分数的分布
plt.figure(figsize=(10, 6))
plt.hist(gt_scores, bins=10, edgecolor='black', color='skyblue')
plt.title('Distribution of Confidence Scores')
plt.xlabel('Confidence Score')
plt.ylabel('Frequency')
plt.grid(True)

# 保存图像到 jpg 文件
output_path = "confidence_score_distribution.jpg"
plt.savefig(output_path, format='jpg', dpi=300)

