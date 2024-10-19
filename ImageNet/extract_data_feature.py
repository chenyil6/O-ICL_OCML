from imagenet_dataset import ImageNetDataset
from transformers import CLIPModel, AutoProcessor, AutoTokenizer
import os
import torch
import pickle
from tqdm import tqdm

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# load clip model and processor
embedding_model = CLIPModel.from_pretrained(
    '/home/chy63/.cache/huggingface/hub/models--clip-vit-base-patch32', 
    local_files_only=True
).to(device)

image_processor = AutoProcessor.from_pretrained(
    '/home/chy63/.cache/huggingface/hub/models--clip-vit-base-patch32', 
    local_files_only=True
)

text_tokenizer = AutoTokenizer.from_pretrained(
    '/home/chy63/.cache/huggingface/hub/models--clip-vit-base-patch32', 
    local_files_only=True
)


# initialize ImageNetDataset
train_dataset = ImageNetDataset(
    root=os.path.join("/data/hyh/imagenet/data", "train"),
    index_file="./imagenet_class_indices.pkl"  # 指定索引文件路径
)

def get_embedding(image):
    inputs = image_processor(images=image, return_tensors="pt").to(device) 
    with torch.no_grad():
        image_features = embedding_model.get_image_features(**inputs)
    return image_features

def get_text_embedding(label):
    inputs = text_tokenizer(text=label, padding=True,return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = embedding_model.get_text_features(**inputs)
    return text_features

features_data = {}

def preprocess_train(sample):
    idx = sample["id"]
    image = sample["image"]
    label = sample["class_name"]

    image_embed = get_embedding(image).squeeze().cpu()

    label_embed = get_text_embedding(label).squeeze().cpu()

    quality = torch.cosine_similarity(image_embed.unsqueeze(0), label_embed.unsqueeze(0), dim=1).item()

    features_data[idx] = [image_embed, quality]


use_train_data = []
M = 10000

for class_id in range(1000):
    data_list = train_dataset.get_data_list_by_class(class_id=class_id)
    use_train_data.extend(data_list)  

print(f"the len of use_train_data(support set + data stream) is {len(use_train_data)}")

# 预处理支持集
while use_train_data:
    sample = use_train_data.pop(0)
    preprocess_train(sample)

with open('./train_idx2embed_quality.pkl', 'wb') as f:
    pickle.dump(features_data, f)

print("train data features are stored...")

test_dataset = ImageNetDataset(os.path.join("/data/hyh/imagenet/data", "val"))
print(f"the len of test_data is {len(test_dataset)}")

features_data_val = {}
def preprocess_val(sample):
    idx = sample["id"]
    image = sample["image"]
    label = sample["class_name"]
    image_embed = get_embedding(image).squeeze().cpu()
    features_data_val[idx] = image_embed

for data in tqdm(test_dataset, desc="Processing test dataset"):
    preprocess_val(data)

with open('./val_idx2embed_quality.pkl', 'wb') as f:
    pickle.dump(features_data_val, f)

print("val data features are stored...")