import numpy as np
import json
from collections import defaultdict
from eval_datasets import VQADataset

train_captions = json.load(open("/data/share/chy/vizwiz/train_questions_vqa_format.json", 'r'))["questions"]
train_answers = json.load(open("/data/share/chy/vizwiz/train_annotations_vqa_format.json", 'r'))["annotations"]
# key 是 question_id , val 是：[[273729001, 0.0, 40973], [153074017, 0.0, 77918], [292597002, 0.0, 107758],...]
qq_dict = np.load("train_question_question_vizwiz.npy", allow_pickle=True).item()

train_image_dir_path = "/data/share/chy/vizwiz/image/train/"
train_questions_json_path = "/data/share/chy/vizwiz/train_questions_vqa_format.json"
train_annotations_json_path = "/data/share/chy/vizwiz/train_annotations_vqa_format.json"
dataset_name = "vizwiz"

full_dataset = VQADataset(
        image_dir_path=train_image_dir_path,
        question_path=train_questions_json_path,
        annotations_path=train_annotations_json_path,
        is_train=True,
        dataset_name=dataset_name,
    )

train_ids = [] # question_id 的 list

for idx, caption in enumerate(train_captions):
    # 把  训练集数据 所有的 question_id 放到 train_id 数组
    train_ids.append(caption["question_id"])

# 遍历 train_ids，弄一个新的dict，key 仍然是 question_id，val是 数据集的idx_list
question_id2answer_list = defaultdict(dict)

count = 0
for question_id in train_ids:
    answer_list = []
    for sublist in qq_dict[question_id]:
        idx = sublist[2]
        answer = full_dataset[idx]["answers"][0]
        answer_list.append(answer)

    count +=1
    # 对answer_list去重，并且保留 元素的相对位置
    answer_list = list(dict.fromkeys(answer_list))
    # 最多保留5个备选答案
    if len(answer_list) <= 5:
        question_id2answer_list[question_id] = answer_list
    else:
        question_id2answer_list[question_id] = answer_list[:5]
    if count <= 100:
        print("question_id:",question_id)
        print("对应的 answer_list：",question_id2answer_list[question_id])


output_file_path = "question_id2answer_list_vizwiz.json"

with open(output_file_path, "w") as json_file:
    json.dump(question_id2answer_list, json_file,indent=4)

print(f"Data written to {output_file_path}")


