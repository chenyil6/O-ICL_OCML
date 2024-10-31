from open_flamingo_v2.open_flamingo.src.factory import create_model_and_transforms
import json
from PIL import Image
import requests
import torch

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

demo_image_one = Image.open(
    requests.get(
        "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True
    ).raw
)
preprocessed = image_processor(demo_image_one)

print("preprocessed shape",preprocessed.shape) # preprocessed shape torch.Size([3, 224, 224])

from einops import rearrange

vision_x = rearrange(preprocessed, "c h w -> 1 1 1 c h w")  # (b, T_img, F, C, H, W)
# 检查特征维度
print("Image features shape:", vision_x.shape)  # Image features shape: torch.Size([1, 1, 1, 3, 224, 224])

vision_x = vision_x.to(device).half()

print("Image features shape:", vision_x.shape)   # Image features shape: torch.Size([1, 1, 1, 3, 224, 224])

# use OpenFlamingo to extract features of an image
assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
b, T, F = vision_x.shape[:3]
assert F == 1, "Only single frame supported"
print("Image features shape:", vision_x.shape)  # Image features shape: torch.Size([1, 1, 1, 3, 224, 224])
vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
print("Image features shape:", vision_x.shape)   # Image features shape: torch.Size([1, 3, 224, 224])


with torch.no_grad():
    vision_x = model.vision_encoder(vision_x)[1]
print("Image features shape:", vision_x.shape)  # Image features shape: torch.Size([1, 256, 1024])

vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
print("Image features shape:", vision_x.shape) # Image features shape: torch.Size([1, 1, 1, 256, 1024])

feature_256_1024 = vision_x.squeeze(0).squeeze(0).squeeze(0).cpu()

vision_x = model.perceiver(vision_x)
print("Image features shape:", vision_x.shape)  # Image features shape: torch.Size([1, 1, 64, 1024])

feature_64_1024 = vision_x.squeeze(0).squeeze(0).cpu()