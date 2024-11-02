import json
import re

# 加载 JSON 文件
with open("./result/open_flamingo-balanced-Online_ICL-M=500-select_strategy=cosine-batch_size=4-update_strategy=fixed_gradient-gradient=0.2-shot=4.json", "r") as file:
    data = json.load(file)

# 初始化计数器
total_count = 0
vote_correct_count = 0
correct_count = 0

# 遍历每个字典元素
for item in data:
    total_count += 1
    
    # 提取预测标签和实际标签
    pred_label = item["pred_label"]
    gt_label = item["gt_label"]

    # 提取 `prompt_text` 中的所有类别标签
    prompt_text = item["prompt_text"]
    prompt_labels = re.findall(r"Output:(.*?)<\|endofchunk\|>", prompt_text)

    # 计算 `pred_label` 出现在 `prompt_labels` 中的次数
    pred_label_count = prompt_labels.count(pred_label)
    
    # 如果 `pred_label` 出现次数大于等于 2，则投票正确
    if pred_label_count >= 2:
        vote_correct_count += 1
    
    # 如果 `pred_label` 等于 `gt_label`，则表示预测正确
    if pred_label == gt_label:
        correct_count += 1

# 计算投票正确率和总正确率
vote_accuracy = vote_correct_count / total_count
accuracy = correct_count / total_count

# 输出结果
print(f"总计数量 (Total Count): {total_count}")
print(f"投票正确率 (Vote Accuracy): {vote_accuracy:.2%}")
print(f"正确率 (Accuracy): {accuracy:.2%}")
