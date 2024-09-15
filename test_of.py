from open_flamingo_4.open_flamingo.src.factory import create_model_and_transforms
import torch

# 设置设备为CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="/data/share/mpt-7b/",
    tokenizer_path="/data/share/mpt-7b/",
    cross_attn_every_n_layers=4,
)


checkpoint_path = "/data/share/OpenFlamingo-9B-vitl-mpt7b/checkpoint.pt"
model.load_state_dict(torch.load(checkpoint_path), strict=False)


from PIL import Image
import requests
import torch

"""
Step 1: Load images
"""
demo_image_one = Image.open(
    requests.get(
        "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True
    ).raw
)

demo_image_two = Image.open(
    requests.get(
        "http://images.cocodataset.org/test-stuff2017/000000028137.jpg",
        stream=True
    ).raw
)

query_image = Image.open(
    requests.get(
        "http://images.cocodataset.org/test-stuff2017/000000028352.jpg",
        stream=True
    ).raw
)


"""
Step 2: Preprocessing images
Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
 batch_size x num_media x num_frames x channels x height x width. 
 In this case batch_size = 1, num_media = 3, num_frames = 1,
 channels = 3, height = 224, width = 224.
"""
vision_x = [image_processor(demo_image_one).unsqueeze(0), image_processor(demo_image_two).unsqueeze(0), image_processor(query_image).unsqueeze(0)]
vision_x = torch.cat(vision_x, dim=0)
vision_x = vision_x.unsqueeze(1).unsqueeze(0)

"""
Step 3: Preprocessing text
Details: In the text we expect an <image> special token to indicate where an image is.
 We also expect an <|endofchunk|> special token to indicate the end of the text 
 portion associated with an image.
"""
tokenizer.padding_side = "left" # For generation padding tokens should be on the left
lang_x = tokenizer(
    ["<image>An image of two cats.<|endofchunk|><image>An image of a bathroom sink.<|endofchunk|><image>An image of"],
    return_tensors="pt",
)


"""
Step 4: Generate text
"""
output = model.generate(
    vision_x=vision_x,
    lang_x=lang_x["input_ids"],
    attention_mask=lang_x["attention_mask"],
    max_new_tokens=20,
    num_beams=1,
    output_scores=True,
    return_dict_in_generate=True
)

initial_length = lang_x["input_ids"].shape[1]

# 提取新生成的 tokens
new_tokens = output["sequences"][:, initial_length:]

print("Newly generated tokens:", new_tokens)

# 解码新生成的 tokens
generated_new_text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

print("Generated new text:", generated_new_text)

print("output['sequences']",output["sequences"])

print("output['sequences'] shape",output["sequences"].shape)

generated_text = tokenizer.batch_decode(output["sequences"], skip_special_tokens=True)

print("generated_text:",generated_text)


for i in output.scores:
    print("Generated scores shape: ", i.shape)


