import torch
from transformers import  AutoProcessor,IdeficsForVisionText2Text

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = "/data/share/pyz/model_weight/idefics-9b"
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
processor = AutoProcessor.from_pretrained(checkpoint)


prompt = ["User:",
           "https://images.unsplash.com/photo-1543349689-9a4d426bee8e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3501&q=80",
           "Classify the following image into a single category.\nAssistant: vegetables.\n",
           "User:",
           "/data/hyh/imagenet/val/n01440764/ILSVRC2012_val_00000293.JPEG",
           "Classify the following image into a single category.\nAssistant:"
           ]


inputs = processor(prompt, return_tensors="pt").to("cuda")
bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

generated_ids = model.generate(**inputs, max_new_tokens=30, bad_words_ids=bad_words_ids)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_text[0])