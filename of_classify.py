from open_flamingo_4.open_flamingo.src.factory import create_model_and_transforms
import torch
import sys
sys.path.append('/data/chy/Online_ICL')
from ImageNet.classification_utils import IMAGENET_1K_CLASS_ID_TO_LABEL,IMAGENET_CLASSNAMES
import math

device_set = 'cuda:1'
device = torch.device(device_set)

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="/data/share/mpt-7b/",
    tokenizer_path="/data/share/mpt-7b/",
    cross_attn_every_n_layers=4,
    device = device_set,
    inference=True,
    precision="fp16",
    checkpoint_path ="/data/share/OpenFlamingo-9B-vitl-mpt7b/checkpoint.pt"
)

print("loading the model successfully....")

from PIL import Image
import requests
import torch

"""
Step 1: Load images
"""
demo_image_one = Image.open("/data/hyh/imagenet/data/val/n01440764/ILSVRC2012_val_00000293.JPEG")

demo_image_two = Image.open("/data/hyh/imagenet/data/val/n01443537/ILSVRC2012_val_00000236.JPEG")

query_image = Image.open("/data/hyh/imagenet/data/val/n01484850/ILSVRC2012_val_00002338.JPEG")

vision_x = [image_processor(demo_image_one).unsqueeze(0), image_processor(demo_image_two).unsqueeze(0), image_processor(query_image).unsqueeze(0)]
vision_x = torch.cat(vision_x, dim=0)
vision_x = vision_x.unsqueeze(1).unsqueeze(0)
vision_x = vision_x.to(device).half()  # 移动到设备上

tokenizer.padding_side = "left" # For generation padding tokens should be on the left
lang_x = tokenizer(
    ["<image>Output:tench.<|endofchunk|><image>Output:gold fish.<|endofchunk|><image>Output:"],
    return_tensors="pt",
)
# 将语言输入移动到设备上
lang_x = {k: v.to(device) for k, v in lang_x.items()}

outputs = model.generate(
    vision_x=vision_x,
    lang_x=lang_x["input_ids"],
    attention_mask=lang_x["attention_mask"],
    min_new_tokens=1,
    max_new_tokens=20,
    num_beams=1,
    length_penalty=1.0,
    output_scores=True,
    return_dict_in_generate=True
)

initial_length = lang_x["input_ids"].shape[1]

# 提取新生成的 tokens
new_tokens = outputs["sequences"][:, initial_length:]

print("Newly generated tokens:", new_tokens)

# 解码新生成的 tokens
generated_new_text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

print("Generated new text:", generated_new_text)

print("output['sequences']",outputs["sequences"])

print("output['sequences'] shape",outputs["sequences"].shape)

generated_text = tokenizer.batch_decode(outputs["sequences"], skip_special_tokens=True)

print("generated_text:",generated_text)


classnames_tokens = tokenizer(IMAGENET_CLASSNAMES)["input_ids"]

overall_log_probs = torch.zeros(len(classnames_tokens))
entropies = torch.zeros(len(classnames_tokens))

for idx,ct in enumerate(classnames_tokens):
    classname_tokens_num = len(ct)
    log_prob = 0
    entropy = 0
    valid = True  # 用于标记生成的token是否足够覆盖类别名称
    for i in range(classname_tokens_num):
        try:
            #  Compute log probabilities and probabilities
            log_scores = torch.nn.functional.log_softmax(outputs.scores[i], dim=-1).squeeze().tolist()
            softmax_scores = torch.nn.functional.softmax(outputs.scores[i], dim=-1).squeeze().tolist()
            # Sum the log probabilities instead of multiplying probabilities
            log_prob += log_scores[ct[i]]
            # Calculate entropy
            entropy += -softmax_scores[ct[i]] * log_scores[ct[i]]
        except IndexError as e:
            print(f"IndexError encountered at position {i} with ct[i]={ct[i]} and token={ct[i]}: {str(e)}")
            log_prob = -float('inf')
            entropy = float('inf')
            valid = False
            break  # Exit the loop if there's an IndexError
    
    if valid:
            log_prob /= classname_tokens_num  # 归一化处理
            entropy /= classname_tokens_num
            overall_log_probs[idx] = torch.exp(torch.tensor(log_prob))   # 将log概率转换为概率
            entropies[idx] = entropy


# 归一化类别概率
overall_log_probs = overall_log_probs / overall_log_probs.sum()
    
# 计算所有类别的总熵
total_entropy = torch.sum(entropies).item()


print("total_entropy :", total_entropy)
