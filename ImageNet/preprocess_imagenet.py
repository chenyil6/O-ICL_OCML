import os
from imagenet_dataset import ImageNetDataset
from classification_utils import IMAGENET_1K_CLASS_ID_TO_LABEL

def preprocess_and_save_index(data_root, output_file):
    # 使用 ImageNetDataset 类来扫描数据并保存索引
    dataset = ImageNetDataset(root=data_root)
    dataset.save_class_index(output_file)
    print("Preprocessing complete. Index file saved.")

if __name__ == "__main__":
    # 设置数据路径和输出路径
    data_root = os.path.join("/data/hyh/imagenet/data", "train")  # 训练集路径
    output_file = "imagenet_class_indices.pkl"  # 索引文件输出路径

    # 生成并保存索引
    preprocess_and_save_index(data_root, output_file)
