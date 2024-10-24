import matplotlib.pyplot as plt
import json

# 绘制单个类的错误率变化图，支持自定义间隔选择中间的点
def plot_error_rate_for_class(error_rate, class_label, num_points=10):
    error_rates = error_rate[class_label]  # 获取该类的错误率历史
    total_points = len(error_rates)

    # 第一个和最后一个元素必须包含，其他的根据 num_points 进行间隔选取
    if num_points > total_points:
        num_points = total_points
    
    indices = [0]  # 确保第 0 个元素被选中
    if num_points > 2:
        # 根据选定的间隔选择中间点
        step = (total_points - 1) // (num_points - 1)  # 计算间隔步长
        indices += list(range(step, total_points - 1, step))[:num_points-2]
    
    indices.append(total_points - 1)  # 确保最后一个元素被选中

    selected_error_rates = [error_rates[i] for i in indices]  # 选择对应的错误率
    inference_times = indices  # 推理次数直接对应错误率列表的索引

    # 绘制错误率变化曲线
    plt.plot(inference_times, selected_error_rates, label=class_label)
    plt.xlabel('Inference times')  # x 轴标签
    plt.ylabel('Error rate')  # y 轴标签
    plt.legend()  # 显示图例
    plt.savefig(f'./visualize/{class_label}_error_rate.jpg')  # 保存图片到指定目录
    plt.show()

# 读取存储的错误率数据
with open("./error_rate_gradient_prototype-gradientUpdate-alpha=0.json", "r") as f:
    error_rate = json.load(f)

# 绘制指定类的错误率变化图
plot_error_rate_for_class(error_rate, "water snake", num_points=20)
