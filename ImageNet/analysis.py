import json
import matplotlib.pyplot as plt

# 文件路径
file_path = "./result/online-open_flamingo-balanced-stream=random-Online_ICL-M=500-select_strategy=topk-update_strategy=prototype.json"

# 读取 JSON 文件
with open(file_path, "r") as f:
    data_json = json.load(f)

correct = 0
total = 0
interval = 1000
accuracy_list = []

# 遍历数据，每隔 250 次计算一次准确率
for data in data_json:
    total += 1
    gt_label = data["gt_label"]
    pred_label = data["pred_label"]
    
    if gt_label == pred_label:
        correct += 1
    
    # 每 250 次计算一次准确率
    if total % interval == 0:
        accuracy = correct / total
        accuracy_list.append(accuracy)
        correct = 0
        total = 0

# 可视化准确率
plt.plot(range(1, len(accuracy_list) + 1), accuracy_list, marker='o')
plt.title('Accuracy Trend Over Time')
plt.xlabel('Interval (2000 updates per interval)')
plt.ylabel('Accuracy')
plt.grid(True)

# 保存图片为 jpg 文件
plt.savefig('balanced_data_random_stream_updates_2000_process.jpg')

# 显示图片（可选）
# plt.show()
