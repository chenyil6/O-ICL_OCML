import torch
from transformers import  AutoProcessor,IdeficsForVisionText2Text
from PIL import Image
import requests
from ImageNet.classification_utils import IMAGENET_1K_CLASS_ID_TO_LABEL,IMAGENET_CLASSNAMES_100


device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = "/data1/pyz/model_weight/idefics-9b"
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
processor = AutoProcessor.from_pretrained(checkpoint)
model.eval()
# 检查是否存在 `_encode_vision_x` 方法
if hasattr(model, '_encode_vision_x'):
    print("The model has the '_encode_vision_x' function.")
else:
    print("The model does NOT have the '_encode_vision_x' function.")

# 可选：打印出模型的所有方法和属性，便于进一步检查
print("Model methods and attributes:")
print(dir(model))


demo_image_one = Image.open("/data/hyh/imagenet/data/val/n01440764/ILSVRC2012_val_00000293.JPEG")

query_image = Image.open("/data/hyh/imagenet/data/val/n01443537/ILSVRC2012_val_00000236.JPEG")

demo_image_two = Image.open("/data/hyh/imagenet/data/val/n01484850/ILSVRC2012_val_00002338.JPEG")

def get_context_images(image_processor, in_context_samples, num_shots=3):
    if num_shots > 0:
        context_images = [
            image_processor(s).unsqueeze(0) for s in in_context_samples
        ]
        context_images = torch.cat(context_images, dim=0)
        context_images = context_images.unsqueeze(1).unsqueeze(0)
    else:
        context_images = None
    return context_images


context_images = get_context_images(processor.image_processor,[demo_image_one,query_image,demo_image_two])

batch_images = context_images.to(device).half()
model._encode_vision_x(vision_x=batch_images)


prompt = ["User:",
           demo_image_one,
           "category: tench.\n",
           "User:",
           demo_image_two,
           "category:great white shark.\n"
           "User:",
           query_image,
           "category:"
           ]

inputs = processor(prompt, return_tensors="pt").to("cuda")
bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

outputs = model.generate(**inputs, max_new_tokens=20, bad_words_ids=bad_words_ids,num_beams=1,length_penalty=1.0,
    output_scores=True,
    return_dict_in_generate=True)


classnames_tokens = processor.tokenizer(IMAGENET_CLASSNAMES_100)["input_ids"]


def get_topk_classifications(outputs, classnames_tokens, topk=2):
    overall_log_probs = torch.zeros(len(classnames_tokens))
    for idx,ct in enumerate(classnames_tokens):
        classname_tokens_num = len(ct)
        log_prob = 0
        valid = True  # 用于标记生成的token是否足够覆盖类别名称
        for i in range(classname_tokens_num):
            try:
                #  Compute log probabilities and probabilities
                log_scores = torch.nn.functional.log_softmax(outputs.scores[i], dim=-1).squeeze().tolist()
                # Sum the log probabilities instead of multiplying probabilities
                log_prob += log_scores[ct[i]]
            except IndexError as e:
                print(f"IndexError encountered at position {i} with ct[i]={ct[i]} and token={ct[i]}: {str(e)}")
                log_prob = -float('inf')
                valid = False
                break  # Exit the loop if there's an IndexError
        
        if valid:
                log_prob /= classname_tokens_num  # 归一化处理
                overall_log_probs[idx] = torch.exp(torch.tensor(log_prob))   # 将log概率转换为概率

    # 归一化类别概率
    overall_log_probs = overall_log_probs / overall_log_probs.sum()

    predicted_classnames, predicted_probs = get_predicted_classname(
        overall_log_probs,
        k=topk,
        class_id_to_name=IMAGENET_1K_CLASS_ID_TO_LABEL,
    )
    return predicted_classnames, predicted_probs, overall_log_probs

def get_predicted_classname(logprobs, k, class_id_to_name):
    """
        Args:
            - logprobs: list containing logprobs for each classname
            - k: number for top-k
            - class_id_to_name: dict mapping class index to classname

        Returns:
            - top-k predicted classnames list type str
            - top-k logprobs list type float
        """
    values, indices = torch.topk(logprobs, k=k, dim=0)  # shape (k,)

    predicted_classnames = [class_id_to_name[ix.item()] for ix in indices]
    predicted_logprobs = values.tolist()

    return predicted_classnames, predicted_logprobs

predicted_classnames, predicted_probs, probs_tensor = get_topk_classifications(outputs,classnames_tokens)

print("predicted_classnames:",predicted_classnames)
print("predicted_probs:",predicted_probs)
print("probs_tensor:",probs_tensor)