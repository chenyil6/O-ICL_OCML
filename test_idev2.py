import requests
import torch
from PIL import Image
from io import BytesIO
from ImageNet.classification_utils import IMAGENET_1K_CLASS_ID_TO_LABEL,IMAGENET_CLASSNAMES_100

from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

# Note that passing the image urls (instead of the actual pil images) to the processor is also possible
demo_image_one = Image.open("/data/hyh/imagenet/data/val/n01440764/ILSVRC2012_val_00000293.JPEG") # tench

demo_image_two = Image.open("/data/hyh/imagenet/data/val/n01443537/ILSVRC2012_val_00000236.JPEG") # gold fish

query_image = Image.open("/data/hyh/imagenet/data/val/n01484850/ILSVRC2012_val_00002338.JPEG") # great_white_shark



processor = AutoProcessor.from_pretrained("/data1/pyz/model_weight/idefics2-8b-base")
model = AutoModelForVision2Seq.from_pretrained("/data1/pyz/model_weight/idefics2-8b-base", device_map="auto")

BAD_WORDS_IDS = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
EOS_WORDS_IDS = [processor.tokenizer.eos_token_id]

# Create inputs
prompts = [
    "<image> category:tench.<image>category:goldfish.<image> category:",  
]
images = [demo_image_one, demo_image_two,query_image]

inputs = processor(images=images, text=prompts, padding=True, truncation=True, return_tensors="pt").to("cuda")


outputs = model.generate(**inputs, bad_words_ids=BAD_WORDS_IDS,min_new_tokens=1, max_new_tokens=20,num_beams=1,
    length_penalty=1.0,
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
                #print(f"IndexError encountered at position {i} with ct[i]={ct[i]} and token={ct[i]}: {str(e)}")
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