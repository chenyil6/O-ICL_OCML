import torch
from transformers import  AutoProcessor,IdeficsForVisionText2Text
from classification_utils import IMAGENET_1K_CLASS_ID_TO_LABEL,IMAGENET_CLASSNAMES
from utils import get_predicted_classname
import math

device = "cuda:0" if torch.cuda.is_available() else "cpu"

checkpoint = "/data/share/pyz/model_weight/idefics-9b"
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
processor = AutoProcessor.from_pretrained(checkpoint)


prompt = ["User:",
    "/data/hyh/imagenet/data/val/n01440764/ILSVRC2012_val_00000293.JPEG",
           "Clarify this image.\nAssistant:tench.\n",
            "User:",
           "/data/hyh/imagenet/data/val/n01443537/ILSVRC2012_val_00000236.JPEG",
           "Clarify this image.\nAssistant:gold fish.\n",
            "User:",
            "/data/hyh/imagenet/data/val/n01484850/ILSVRC2012_val_00002338.JPEG",
           "Clarify this image.\nAssistant:",
           ]


inputs = processor(prompt, return_tensors="pt").to("cuda:0")
bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

outputs = model.generate(**inputs, max_new_tokens=20, bad_words_ids=bad_words_ids,output_scores=True,
    return_dict_in_generate=True)
generated_text = processor.tokenizer.batch_decode(outputs["sequences"], skip_special_tokens=True)

print("generated_text:",generated_text)
classnames_tokens = processor.tokenizer(IMAGENET_CLASSNAMES)["input_ids"]
print(f"len(classnames_tokens):",len(classnames_tokens))

overall_log_probs = []
print("len(outputs.scores):",len(outputs.scores))

for idx,ct in enumerate(classnames_tokens):
    classname_tokens_num = len(ct)
    log_prob = 0
    for i in range(classname_tokens_num):
        try:
            if i >= len(outputs.scores):
                raise IndexError("Trying to access scores beyond the generated tokens")

            # Compute log probabilities instead of probabilities
            log_scores = torch.nn.functional.log_softmax(outputs.scores[i], dim=-1).squeeze().tolist()

            if ct[i] >= len(log_scores):
                raise IndexError(f"Token {ct[i]} is out of bounds for the log_scores of length {len(log_scores)}")

            # Sum the log probabilities instead of multiplying probabilities
            print("log_scores[ct[i]]:", log_scores[ct[i]])
            log_prob += log_scores[ct[i]]
        except IndexError as e:
            print(f"IndexError encountered at position {i} with ct[i]={ct[i]} and token={ct[i]}: {str(e)}")
            log_prob = -float('inf')
            break  # Exit the loop if there's an IndexError
    print("log_prob:", log_prob)
    log_prob /= classname_tokens_num
    print(f"log_prob for {IMAGENET_CLASSNAMES[idx]} is {log_prob}")
    overall_log_probs.append(log_prob) # list , len is  1000
print("len(overall_log_probs):",len(overall_log_probs))


# 计算平均值
filtered_log_probs = [log_prob for log_prob in overall_log_probs if math.isfinite(log_prob)]

# 如果 filtered_log_probs 为空，则说明所有值都是 -inf，可以处理这种情况
if len(filtered_log_probs) > 0:
    average_log_prob = sum(filtered_log_probs) / len(filtered_log_probs)
else:
    average_log_prob = float('-inf')  # 或者根据你的需求处理这个情况
print("Average log probability:", average_log_prob)

predicted_classnames, predicted_logprobs = get_predicted_classname(
            overall_log_probs,
            k=1,
            class_id_to_name=IMAGENET_1K_CLASS_ID_TO_LABEL,
        )

print("predicted_logprobs[0]:",predicted_logprobs[0])

# 计算相对概率分数
relative_score = torch.exp(torch.tensor(predicted_logprobs[0]  - average_log_prob)).item()
print("Relative Score:", relative_score)



print("predicted results:",predicted_classnames[0])

