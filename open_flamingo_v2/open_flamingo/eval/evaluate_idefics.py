import argparse
import importlib
import json
import os
import random
import uuid
from collections import defaultdict
import copy

import nltk

from einops import repeat
import more_itertools
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from coco_metric import compute_cider, postprocess_captioning_generation
from transformers import IdeficsForVisionText2Text, AutoProcessor

from tqdm import tqdm

import sys


from eval_datasets import VQADataset

from classification_utils import (
    IMAGENET_CLASSNAMES,
    IMAGENET_1K_CLASS_ID_TO_LABEL,
    HM_CLASSNAMES,
    HM_CLASS_ID_TO_LABEL,
)

from eval_model import BaseEvalModel

print(sys.path)
sys.path.append("/data/chy/openflamingo_v2/")
print(sys.path)

from ok_vqa_utils import postprocess_ok_vqa_generation
from open_flamingo_chy.src.flamingo import Flamingo

from vqa_metric import compute_vqa_accuracy, postprocess_vqa_generation
from vqa_metric_vqacp import compute_vqacp_accuracy
from open_flamingo_chy.train.distributed import init_distributed_device, world_info_from_env

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    help="Model name. Currently only `OpenFlamingo` is supported.",
    default="open_flamingo",
)
parser.add_argument(
    "--results_file", type=str, default=None, help="JSON file to save results"
)

# Trial arguments
parser.add_argument("--shots", nargs="+", default=[0, 4, 8, 16, 32], type=int)
parser.add_argument(
    "--num_trials",
    type=int,
    default=1,
    help="Number of trials to run for each shot using different demonstrations",
)
parser.add_argument(
    "--trial_seeds",
    nargs="+",
    type=int,
    default=[42],
    help="Seeds to use for each trial for picking demonstrations and eval sets",
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=-1,
    help="Number of samples to evaluate on. -1 for all samples.",
)
parser.add_argument(
    "--query_set_size", type=int, default=2048, help="Size of demonstration query set"
)

parser.add_argument("--batch_size", type=int, default=8)

parser.add_argument(
    "--no_caching_for_classification",
    action="store_true",
    help="Use key-value caching for classification evals to speed it up. Currently this doesn't underperforms for MPT models.",
)

# Per-dataset evaluation flags

parser.add_argument(
    "--eval_vizwiz",
    action="store_true",
    default=False,
    help="Whether to evaluate on VizWiz.",
)

parser.add_argument(
    "--eval_ok_vqa",
    action="store_true",
    default=False,
    help="Whether to evaluate on OK-VQA.",
)

parser.add_argument(
    "--eval_vqav2",
    action="store_true",
    default=False,
    help="Whether to evaluate on VQAV2.",
)

parser.add_argument(
    "--eval_vqa_cp",
    action="store_true",
    default=False,
    help="Whether to evaluate on VQAV2-cp.",
)


# Dataset arguments
## OK-VQA Dataset
parser.add_argument(
    "--ok_vqa_train_image_dir_path",
    type=str,
    help="Path to the vqav2/train2014 directory.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_train_questions_json_path",
    type=str,
    help="Path to the v2_OpenEnded_mscoco_train2014_questions.json file.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_train_annotations_json_path",
    type=str,
    help="Path to the v2_mscoco_train2014_annotations.json file.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_test_image_dir_path",
    type=str,
    help="Path to the vqav2/val2014 directory.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_test_questions_json_path",
    type=str,
    help="Path to the v2_OpenEnded_mscoco_val2014_questions.json file.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_test_annotations_json_path",
    type=str,
    help="Path to the v2_mscoco_val2014_annotations.json file.",
    default=None,
)

## VizWiz Dataset
parser.add_argument(
    "--vizwiz_train_image_dir_path",
    type=str,
    help="Path to the vizwiz train images directory.",
    default=None,
)
parser.add_argument(
    "--vizwiz_test_image_dir_path",
    type=str,
    help="Path to the vizwiz test images directory.",
    default=None,
)
parser.add_argument(
    "--vizwiz_train_questions_json_path",
    type=str,
    help="Path to the vizwiz questions json file.",
    default=None,
)
parser.add_argument(
    "--vizwiz_train_annotations_json_path",
    type=str,
    help="Path to the vizwiz annotations json file.",
    default=None,
)
parser.add_argument(
    "--vizwiz_test_questions_json_path",
    type=str,
    help="Path to the vizwiz questions json file.",
    default=None,
)
parser.add_argument(
    "--vizwiz_test_annotations_json_path",
    type=str,
    help="Path to the vizwiz annotations json file.",
    default=None,
)

## VQAV2 Dataset
parser.add_argument(
    "--vqav2_train_image_dir_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_train_questions_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_train_annotations_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_test_image_dir_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_test_questions_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_test_annotations_json_path",
    type=str,
    default=None,
)

## VQA-CP Dataset
parser.add_argument(
    "--vqa_cp_train_image_dir_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqa_cp_train_questions_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqa_cp_train_annotations_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqa_cp_test_image_dir_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqa_cp_test_questions_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqa_cp_test_annotations_json_path",
    type=str,
    default=None,
)

# Distributed evaluation
parser.add_argument(
    "--dist-url",
    default="env://",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--horovod",
    default=False,
    action="store_true",
    help="Use horovod for distributed training.",
)
parser.add_argument(
    "--no-set-device-rank",
    default=False,
    action="store_true",
    help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
)

import sys
def main():
    args, leftovers = parser.parse_known_args()

    #print(sys.path)
    module = importlib.import_module(f"open_flamingo_chy.eval.models.{args.model}")

    model_args = {
        leftovers[i].lstrip("-"): leftovers[i + 1] for i in range(0, len(leftovers), 2)
    }
    eval_model = module.EvalModel(model_args)

    # set up distributed evaluation
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args, eval_model.device)
    # device_id = "cuda:" + str(model_args["device"])
    print(device_id)
    # device_id = torch.device(device_id)
    eval_model.set_device(device_id)
    eval_model.init_distributed()

    if args.model != "open_flamingo" and args.shots != [0]:
        raise ValueError("Only 0 shot eval is supported for non-open_flamingo models")

    if len(args.trial_seeds) != args.num_trials:
        raise ValueError("Number of trial seeds must be == number of trials.")

    results = defaultdict(list)

    # load model
    checkpoint = "/data1/share/idefics/idefics"
    model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
    processor = AutoProcessor.from_pretrained(checkpoint)

    if args.eval_vizwiz:
        print("Evaluating on VizWiz...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                vizwiz_score = evaluate_vqa(
                    args=args,
                    eval_model=model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="vizwiz",
                )
                if args.rank == 0:
                    print(f"Shots {shot} Trial {trial} VizWiz score: {vizwiz_score}")
                    scores.append(vizwiz_score)

            if args.rank == 0:
                print(f"Shots {shot} Mean VizWiz score: {np.nanmean(scores)}")
                results["vizwiz"].append(
                    {"shots": shot, "trials": scores, "mean": np.nanmean(scores)}
                )

    if args.eval_ok_vqa:
        print("Evaluating on OK-VQA...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                ok_vqa_score = evaluate_vqa(
                    args=args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="ok_vqa",
                )
                if args.rank == 0:
                    print(f"Shots {shot} Trial {trial} OK-VQA score: {ok_vqa_score}")
                    scores.append(ok_vqa_score)

            if args.rank == 0:
                print(f"Shots {shot} Mean OK-VQA score: {np.nanmean(scores)}")
                results["ok_vqa"].append(
                    {"shots": shot, "trials": scores, "mean": np.nanmean(scores)}
                )

    if args.eval_vqav2:
        print("Evaluating on VQAv2...")
        for shot in args.shots:
            scores = []
            print("shot: ", shot)
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                vqa_score = evaluate_vqa(
                    args=args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="vqav2",
                )
                if args.rank == 0:
                    print(f"Shots {shot} Trial {trial} VQA score: {vqa_score}")
                    scores.append(vqa_score)

            if args.rank == 0:
                print(f"Shots {shot} Mean VQA score: {np.nanmean(scores)}")
                results["vqav2"].append(
                    {"shots": shot, "trials": scores, "mean": np.nanmean(scores)}
                )

    if args.eval_vqa_cp:
        print("Evaluating on VQA-CP...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                vqa_cp_score = evaluate_vqa(
                    args=args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="vqa_cp",
                )
                if args.rank == 0:
                    print(f"Shots {shot} Trial {trial} OK-VQA score: {vqa_cp_score}")
                    scores.append(vqa_cp_score)

            if args.rank == 0:
                print(f"Shots {shot} Mean VQA-CP score: {np.nanmean(scores)}")
                results["vqa_cp"].append(
                    {"shots": shot, "trials": scores, "mean": np.nanmean(scores)}
                )

    if args.rank == 0 and args.results_file is not None:
        with open(args.results_file, "w") as f:
            json.dump(results, f,indent=4)



def get_random_indices(num_samples, query_set_size, full_dataset, seed):
    if num_samples + query_set_size > len(full_dataset):
        raise ValueError(
            f"num_samples + query_set_size must be less than {len(full_dataset)}"
        )

    # get a random subset of the dataset
    np.random.seed(seed)
    random_indices = np.random.choice(
        len(full_dataset), num_samples + query_set_size, replace=False
    )
    return random_indices

def get_query_set(subset_train_dataset, query_set_size, seed):
    np.random.seed(seed)
    query_set = np.random.choice(len(subset_train_dataset), query_set_size, replace=False)
    return [subset_train_dataset[i] for i in query_set]


def prepare_sub_train_dataset(train_dataset, indices_list):
    sub_dataset = torch.utils.data.Subset(train_dataset, indices_list)
    return sub_dataset


# 准备从test_dataset中取出 num_samples 条数据
# 其中由很多批次组成，一个批次大小是batch_size，batch * batch_size = num_samples
def prepare_eval_samples(test_dataset, num_samples, batch_size, seed):
    if len(test_dataset)<num_samples:
        num_samples = len(test_dataset)
    np.random.seed(seed)
    random_indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    dataset = torch.utils.data.Subset(test_dataset, random_indices)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=custom_collate_fn,
    )
    return loader

def get_incorrect_answer(answer_list,correct_answer):
    # 获取 condidate_ansers
    candidate_answers = [answer for answer in answer_list if answer != correct_answer]
    return random.choice(candidate_answers)


def sample_batch_demos_from_query_set(query_set, num_samples, batch, retrieval_type, clip):
    if not clip:
        output = []
        for _ in batch:
            o = []
            a = random.sample(range(len(query_set)), num_samples)
            for j, i in enumerate(a):
                x = copy.deepcopy(query_set[i])
                o.append(x)
            output.append(o)
        return output
    else:
        if retrieval_type == "mix_img_cap":
            return [[query_set.id2item(id) for id in
                     batch["clip_images"][i][:(num_samples // 2)] + batch["clip_captions"][i][:(num_samples // 2)]] for
                    i in range(len(batch["image"]))]
        else:# sqqr/siir/sqaqar
            return [[query_set.id2item(id) for id in batch[retrieval_type][i][:num_samples]] for i in
                    range(len(batch["image"]))]

def compute_effective_num_shots(num_shots, model_type):
    if model_type == "open_flamingo":
        return num_shots if num_shots > 0 else 2
    return num_shots


def custom_collate_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch

# 定义 preprocess 函数，用于标注句子中的词性
def preprocess(text):
    words = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    return pos_tags


def remove_words_by_pos(text, remove_pos):
    pos_tags = preprocess(text)
    new_sentence = []

    for word, pos_tag in pos_tags:
        if pos_tag not in remove_pos:
            new_sentence.append(word)

    result = ' '.join(new_sentence)
    return result


def evaluate_vqa(
    args: argparse.Namespace,
    eval_model: BaseEvalModel,
    seed: int = 42,
    min_generation_length: int = 0,
    max_generation_length: int = 5,
    num_beams: int = 3,
    length_penalty: float = 0.0,
    num_shots: int = 8,
    dataset_name: str = "vqav2",
):
    """
    Evaluate a model on VQA datasets. Currently supports VQA v2.0, OK-VQA, VizWiz and TextVQA.

    Args:
        args (argparse.Namespace): arguments
        eval_model (BaseEvalModel): model to evaluate
        seed (int, optional): random seed. Defaults to 42.
        max_generation_length (int, optional): max generation length. Defaults to 5.
        num_beams (int, optional): number of beams to use for beam search. Defaults to 3.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        dataset_name (string): type of vqa dataset: currently supports vqav2, ok_vqa. Defaults to vqav2.
    Returns:
        float: accuracy score
    """
    if dataset_name == "ok_vqa":
        train_image_dir_path = args.ok_vqa_train_image_dir_path
        train_questions_json_path = args.ok_vqa_train_questions_json_path
        train_annotations_json_path = args.ok_vqa_train_annotations_json_path
        test_image_dir_path = args.ok_vqa_test_image_dir_path
        test_questions_json_path = args.ok_vqa_test_questions_json_path
        test_annotations_json_path = args.ok_vqa_test_annotations_json_path
        question_to_answerList_path = "/data/chy/flamingo/open_flamingo/eval/question_answers.json"
        questionId_to_answerList_path = "/data/chy/openflamingo_v2/open_flamingo_chy/eval/question_id2answer_list_okvqa.json"
    elif dataset_name == "vizwiz":
        train_image_dir_path = args.vizwiz_train_image_dir_path
        train_questions_json_path = args.vizwiz_train_questions_json_path
        train_annotations_json_path = args.vizwiz_train_annotations_json_path
        test_image_dir_path = args.vizwiz_test_image_dir_path
        test_questions_json_path = args.vizwiz_test_questions_json_path
        test_annotations_json_path = args.vizwiz_test_annotations_json_path
        question_to_answerList_path = "/data/chy/flamingo/open_flamingo/eval/question_answers_vizwiz.json"
        questionId_to_answerList_path = "/data/chy/openflamingo_v2/open_flamingo_chy/eval/question_id2answer_list_vizwiz.json"
    elif dataset_name == "vqav2":
        train_image_dir_path = args.vqav2_train_image_dir_path
        train_questions_json_path = args.vqav2_train_questions_json_path
        train_annotations_json_path = args.vqav2_train_annotations_json_path
        test_image_dir_path = args.vqav2_test_image_dir_path
        test_questions_json_path = args.vqav2_test_questions_json_path
        test_annotations_json_path = args.vqav2_test_annotations_json_path
        question_to_answerList_path = "/data/chy/openflamingo_v2/open_flamingo_chy/eval/data.json"
        questionId_to_answerList_path = "/data/chy/flamingo/open_flamingo/eval/question_id2answer_list.json"
    elif dataset_name == "vqa_cp":
        train_image_dir_path = args.vqa_cp_train_image_dir_path
        train_questions_json_path = args.vqa_cp_train_questions_json_path
        train_annotations_json_path = args.vqa_cp_train_annotations_json_path
        test_image_dir_path = args.vqa_cp_test_image_dir_path
        test_questions_json_path = args.vqa_cp_test_questions_json_path
        test_annotations_json_path = args.vqa_cp_test_annotations_json_path
        question_to_answerList_path = "/data/chy/openflamingo_v2/open_flamingo_chy/eval/data.json"
        questionId_to_answerList_path = "/data/chy/flamingo/open_flamingo/eval/question_id2answer_list.json"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_dataset = VQADataset(
        train_image_dir_path=train_image_dir_path,
        val_image_dir_path= test_image_dir_path if dataset_name=="vqa_cp" else None,
        question_path=train_questions_json_path,
        annotations_path=train_annotations_json_path,
        is_train=True,
        dataset_name=dataset_name,
    )

    test_dataset = VQADataset(
        train_image_dir_path=train_image_dir_path,
        val_image_dir_path=test_image_dir_path if dataset_name=="vqa_cp" else None,
        question_path=test_questions_json_path,
        annotations_path=test_annotations_json_path,
        is_train=False,
        dataset_name=dataset_name,
    )

    # declarative sentences
    train_declaration_path = "/data/pjw/project/open_flamingo_v2/declaration/train2014_declarative.json"
    val_declaration_path = "/data/pjw/project/open_flamingo_v2/declaration/val2014_declarative.json"
    if os.path.exists(train_declaration_path):
        with open(train_declaration_path, 'r') as file:
            train_declaration_datasets = json.load(file)
    else:
        print("Train_declaration file not exist")
    if os.path.exists(val_declaration_path):
        with open(val_declaration_path, 'r') as file:
            val_declaration_datasets = json.load(file)
    else:
        print("Val_declaration file not exist")

    # 有效的 shots
    effective_num_shots = compute_effective_num_shots(num_shots, args.model)

    # 从 test_dataset中取 num_samples 条数据，分为num_samples/batch_size个批次
    test_dataloader = prepare_eval_samples(
        test_dataset,
        args.num_samples if args.num_samples > 0 else len(test_dataset),
        args.batch_size,
        seed,
    )

    control_signals = {"clip": False,
                       "retrieval_type": "none", # SIIR;SQQR;SQAQAR;SQAQAR_rs;SQAQAR_SIIR;
                       "instruction_type": "none", # instruction_false; instruction_2; instruction_3;instruction_4
                       "declaration": False,
                       "order": "order", # order; reverse
                       "SQAQAR_type":"normal", # normal;replaceI;replaceQ;
                       "mismatch_type": "none", # answer;image;none;qa
                       "query_question_type":"normal"}    # del_W;del_W_N;normal

    print("control signals of prompt: ", control_signals)

    if control_signals["clip"]:
        random_uuid = "PREDICTION_FILE_{}_{}_declare{}_{}_replace_{}_mismatch_{}".format(control_signals["retrieval_type"],
                                                  num_shots,
                                                  control_signals["declaration"],
                                                  control_signals["order"],
                                                  control_signals["SQAQAR_type"],
                                                  control_signals["mismatch_type"])
    else:
        random_uuid = "PREDICTION_FILE_RS_{}_{}_declare_{}_{}_mismatch_{}_{}".format(num_shots,
                                                  control_signals["instruction_type"],
                                                  control_signals["declaration"],
                                                  control_signals["order"],control_signals["mismatch_type"],
                                                  control_signals["query_question_type"])


    predictions = []

    # 指定目录路径
    directory = "/data/chy/openflamingo_v2/results"

    # 构建完整的文件路径
    prediction_file = os.path.join(directory, f"{dataset_name}_RES_{random_uuid}.json")

    np.random.seed(
        seed + args.rank
    )  # make sure each worker has a different seed for the random context samples
    if not os.path.exists(prediction_file):
        for batch in tqdm(
                test_dataloader,
                desc=f"Running inference {dataset_name}",
                disable=args.rank != 0,
        ):
            batch_demo_samples = sample_batch_demos_from_query_set(
                train_dataset, effective_num_shots, batch,
                control_signals["retrieval_type"], control_signals["clip"])

            prompts = []

            for i in range(len(batch)):
                # 得到 ice
                in_context_samples = batch_demo_samples[i]

                demon = []

                demon.extend(["Instruction: provide an answer to the question. Use the image to answer."+'\n'])

                for ice in in_context_samples:
                    # 向demonstration列表中追加每个元素
                    demon.extend([
                        f"Image: {ice['image']}",
                        f"Question:{ice['question'].strip()} Answer: {ice['answers'][0].strip()}"+'\n'
                    ])
                demon.extend([
                        f"Image:{batch[i]['image']}", # 添加图像
                        f"Question:{batch[i]['question'].strip()} Answer:"])
                prompts.append(demon)

            processor = AutoProcessor.from_pretrained(checkpoint)
            inputs = processor(prompts, return_tensors="pt").to()
            # --single sample mode
            # inputs = processor(prompts[0], return_tensors="pt").to(device)

            generated_ids = model.generate(**inputs, max_length=128)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

            process_function = (
                postprocess_ok_vqa_generation
                if dataset_name == "ok_vqa"
                else postprocess_vqa_generation
            )

            new_predictions = map(process_function, outputs)

            # for new_prediction, sample_id in zip(new_predictions, batch["question_id"]):
            #     predictions.append({"answer": new_prediction, "question_id": sample_id})
            new_predictions = list(new_predictions)
            for i in range(len(batch["image"])):
                predictions.append({"answer": new_predictions[i], "question_id": batch["question_id"][i],
                                    "prompt_text": batch_text[i],
                                    "prompt_images": [img['image_id'] for img in batch_demo_samples[i]],
                                    "prompt_question_id": [qui['question_id'] for qui in batch_demo_samples[i]],
                                    "question": batch["question"][i]
                                    })


        # all gather
        all_predictions = [None] * args.world_size
        torch.distributed.all_gather_object(all_predictions, predictions)  # list of lists
        if args.rank != 0:
            return

        all_predictions = [
            item for sublist in all_predictions for item in sublist
        ]  # flatten

        with open(prediction_file, "w") as f:
            f.write(json.dumps(all_predictions, indent=4))

    if test_annotations_json_path is not None:
        if dataset_name == "vqa_cp":
            acc = compute_vqacp_accuracy(
                prediction_file,
                test_questions_json_path,
                test_annotations_json_path,
            )
        else:
            acc = compute_vqa_accuracy(
                prediction_file,
                test_questions_json_path,
                test_annotations_json_path,
                )
        # delete the temporary file
        #os.remove(f"{dataset_name}results_{random_uuid}.json")

    else:
        print("No annotations provided, skipping accuracy computation.")
        print("Temporary file saved to:", prediction_file)
        acc = None

    return acc




if __name__ == "__main__":
    main()
