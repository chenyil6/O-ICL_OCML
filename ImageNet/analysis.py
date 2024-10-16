import pickle

# 加载索引文件
with open("./imagenet_class_indices.pkl", "rb") as f:
    class_indices = pickle.load(f)

# 打印其中一部分数据进行验证
for class_id, indices in list(class_indices.items())[:5]:  # 打印前5个类的信息
    print(f"Class ID: {class_id}, Number of images: {len(indices)}, Sample indices: {indices[:5]}")
