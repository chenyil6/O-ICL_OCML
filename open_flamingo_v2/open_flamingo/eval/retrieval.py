import torch
import clip
from PIL import Image
import os
import json
from tqdm import tqdm
import numpy as np
from style_split import COCOStyleDataset, FlickrStyleDataset, COCOVqaDataset
from collections import defaultdict

device = "cuda:6" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def extract_features(filenames):
    features = []

    for img in tqdm(filenames):
        imge_input = preprocess(Image.open(img)).unsqueeze(0).to(device)
        with torch.no_grad():
            features.append(model.encode_image(imge_input))
    features = torch.stack(features).squeeze(1)

    print(features.shape)
    return features

train_data_dir = "/data/wyl/coco_data/train2014"
val_data_dir = "/data/wyl/coco_data/val2014"

# prepare train data for vqa_cp
train_captions = json.load(open("/data/share/chy/VQA_CP/vqacp_v2_train_questions.json", 'r'))["questions"]
train_answers = json.load(open("/data/share/chy/VQA_CP/vqacp_v2_train_annotations.json", 'r'))["annotations"]
# extract train feature
train_ids = []
train_img_ids = []
train_filenames = []

train_captions = json.load(open("/data/share/chy/vizwiz/train_questions_vqa_format.json", 'r'))["questions"]

train_ids = [] # question_id 的 list

for idx, caption in enumerate(train_captions):
    # 把  训练集数据 所有的 question_id 放到 train_id 数组
    train_ids.append(caption["question_id"])

# train_text_features_vizwiz.npy
train_text_features = torch.from_numpy(np.load("train_text_features_vizwiz.npy")).to(device)

print(train_text_features.shape)

train_text_features /= train_text_features.norm(dim=-1, keepdim=True)

import faiss

output_all = defaultdict(dict)

# Similarity-based Question-Question Retrieval
def SQQR_retrieval():
    # 使用 Faiss 索引来添加训练集中的文本特征向量，以便后续可以在这些向量上执行相似性搜索
    # Faiss 索引就会在内部构建数据结构，以便能够快速地在这些向量上执行近似最近邻搜索
    index = faiss.IndexFlatL2(512)
    index.add(train_text_features.cpu())
    print(index.ntotal)

    values_caps, indices_caps = index.search(train_text_features.cpu(), 2000)
    np.save("train_question_question_indices_vizwiz.npy", indices_caps)
    np.save("train_question_question_values_vizwiz.npy", values_caps)

    for idx, (value, index) in enumerate(zip(values_caps, indices_caps)):
        output_caps = []
        for id, val in zip(index[:32], value[:32]):
            # [train_ids[id] 是 question_id, id ：在数据集中的idx
            output_caps.append([train_ids[id], val.tolist(), id])
        #output_all[train_ids[idx]]["captions"] = output_caps
        if idx < 10:
            print(f"idx:{idx}, output_caps:{output_caps}")
        output_all[train_ids[idx]] = output_caps

    np.save('train_question_question_vizwiz.npy', output_all)
    print("SQQR_retrieval has write in the  train_question_question_vizwiz.npy ...")

SQQR_retrieval()