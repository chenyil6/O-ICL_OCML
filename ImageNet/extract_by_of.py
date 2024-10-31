import argparse
from inferencers import *
import logging
from transformers import AutoTokenizer, CLIPModel,AutoProcessor
import sys
sys.path.append('/data/chy/online')
from open_flamingo_v2.open_flamingo.src.factory import create_model_and_transforms
from transformers import IdeficsForVisionText2Text, AutoProcessor
import json
from PIL import Image
import requests
from einops import rearrange
from tqdm import tqdm


device_set = "cuda:0"
device = torch.device(device_set)

model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="/data/share/mpt-7b/",
            tokenizer_path="/data/share/mpt-7b/",
            cross_attn_every_n_layers=4,
            precision="fp16",
            inference=True,
            device=device_set,
            checkpoint_path="/data/share/OpenFlamingo-9B-vitl-mpt7b/checkpoint.pt"
        )

def get_embedding(image):
    # 预处理图像
    preprocessed = image_processor(image)
    vision_x = rearrange(preprocessed, "c h w -> 1 1 1 c h w").to(device).half()

    # 提取特征 (256, 1024)
    with torch.no_grad():
        vision_x_flat = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        features_256 = model.vision_encoder(vision_x_flat)[1] # torch.Size([1, 256, 1024])
        features_256 = rearrange(features_256, "(b T F) v d -> b T F v d", b=1, T=1, F=1) # torch.Size([1, 1, 1, 256, 1024])

        # 提取特征 (64, 1024)
        features_64 = model.perceiver(features_256).squeeze(0).squeeze(0) #torch.Size([1, 1, 64, 1024])

        features_256 = features_256.squeeze(0).squeeze(0).squeeze(0)

    return features_256

# 存储特征的字典
train_features_data_256 = {}
#train_features_data_64 = {}


train_dataset = ImageNetDataset(
    root=os.path.join("/data/hyh/imagenet/data", "train"),
    index_file="./imagenet_class_indices.pkl"
)

use_train_data = []

for class_id in range(1000):
    data_list = train_dataset.get_data_list_by_class(class_id=class_id)
    use_train_data.extend(data_list)

print(f"the len of use_train_data(support set + data stream) is {len(use_train_data)}")

# 处理图像并保存特征
def preprocess_train(sample):
    idx = sample["id"]
    image = sample["image"]
    
    # 获取特征
    features_256 = get_embedding(image)
    
    # 存入字典
    train_features_data_256[idx] = features_256.cpu()


# 设置分批处理的参数
batch_size = 50000
num_batches = (len(use_train_data) // batch_size) + 1  # 计算需要的批次数
batch_files = []  # 存储每个文件的路径

# 提取并存储每个样本的特征
for batch_idx in range(num_batches):
    batch_samples = use_train_data[batch_idx * batch_size:(batch_idx + 1) * batch_size]
    for sample in tqdm(batch_samples, desc=f"Processing batch {batch_idx + 1}/{num_batches}"):
        preprocess_train(sample)

    # 将特征存储到不同的 .pkl 文件中
    batch_file_path = f"/data/chy/feacture_cache/train_features_256x1024_batch_{batch_idx + 1}.pkl"
    with open(batch_file_path, "wb") as f:
        pickle.dump(train_features_data_256, f)

    batch_files.append(batch_file_path)  # 添加文件路径到列表
    train_features_data_256.clear()  # 清空字典以释放内存


# 处理图像并保存特征
def preprocess_val(sample):
    idx = sample["id"]
    image = sample["image"]
    
    # 获取特征
    features_256, features_64 = get_embedding(image)
    
    # 存入字典
    val_features_data_256[idx] = features_256.cpu()

# 存储特征的字典
val_features_data_256 = {}

test_dataset = ImageNetDataset(os.path.join("/data/hyh/imagenet/data", "val"))

for data in tqdm(test_dataset, desc="Processing test dataset"):
    preprocess_val(data)

# 将特征存储到不同的 .pkl 文件中
with open("/data/chy/feacture_cache/val_features_256x1024_all.pkl", "wb") as f:
    pickle.dump(val_features_data_256, f)

print("测试集特征提取与存储完成。")

