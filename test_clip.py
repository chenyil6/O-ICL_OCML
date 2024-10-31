from PIL import Image
from transformers import CLIPModel, AutoProcessor, AutoTokenizer
import os
import torch
import pickle
from tqdm import tqdm
import requests

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# load clip model and processor
embedding_model = CLIPModel.from_pretrained(
    '/data1/chy/clip-vit-large-patch14', 
    local_files_only=True
).to(device)

image_processor = AutoProcessor.from_pretrained(
    '/data1/chy/clip-vit-large-patch14', 
    local_files_only=True
)

text_tokenizer = AutoTokenizer.from_pretrained(
    '/data1/chy/clip-vit-large-patch14', 
    local_files_only=True
)


def get_embedding(image):
    inputs = image_processor(images=image, return_tensors="pt").to(device) 
    with torch.no_grad():
        image_features = embedding_model.get_image_features(**inputs)
    return image_features


url ="http://images.cocodataset.org/val2017/000000039769.jpg"

image1 = Image.open(requests.get(url, stream=True).raw)
image2 = Image.open(requests.get(url, stream=True).raw)
images = []
images.append(image1)
images.append(image2)

inputs = image_processor(images=images)
print("inputs:",inputs)
with torch.no_grad():
    image_features = embedding_model.get_image_features(**inputs)

#print(f'Inputs shape: {inputs["pixel_values"].shape}')  # 打印输入张量的形状
print(image_features.shape)