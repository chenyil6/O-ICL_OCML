import argparse
import copy
import importlib
import json
import os
import random
import uuid
from collections import defaultdict

from einops import repeat
import more_itertools
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from coco_metric import compute_cider, postprocess_captioning_generation
from eval_datasets import (
    VQADataset,
    ImageNetDataset,
    HatefulMemesDataset,
)
from tqdm import tqdm


from eval_datasets import VQADataset, ImageNetDataset
from classification_utils import (
    IMAGENET_CLASSNAMES,
    IMAGENET_1K_CLASS_ID_TO_LABEL,
    HM_CLASSNAMES,
    HM_CLASS_ID_TO_LABEL,
)

from eval_model import BaseEvalModel

from ok_vqa_utils import postprocess_ok_vqa_generation
from open_flamingo.open_flamingo.src.flamingo import Flamingo
from vqa_metric import compute_vqa_accuracy, postprocess_vqa_generation

from open_flamingo.open_flamingo.train.distributed import init_distributed_device, world_info_from_env

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
parser.add_argument("--shots", nargs="+", default=[4, 8, 16, 32], type=int)
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
    "--eval_imagenet",
    action="store_true",
    default=False,
    help="Whether to evaluate on ImageNet.",
)

## Imagenet dataset
parser.add_argument("--imagenet_root", type=str, default="/tmp")


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


def main():
    args, leftovers = parser.parse_known_args()
    module = importlib.import_module(f"open_flamingo.open_flamingo.eval.models.{args.model}")

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
    # eval_model.init_distributed()

    # if args.model != "open_flamingo" and args.shots != [0]:
    #     raise ValueError("Only 0 shot eval is supported for non-open_flamingo models")

    if len(args.trial_seeds) != args.num_trials:
        raise ValueError("Number of trial seeds must be == number of trials.")

    results = defaultdict(list)

    if args.eval_imagenet:
        print("Evaluating on ImageNet...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                imagenet_score = evaluate_classification(
                    args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    no_kv_caching=args.no_caching_for_classification,
                    dataset_name="imagenet",
                )
                if args.rank == 0:
                    print(
                        f"Shots {shot} Trial {trial} "
                        f"ImageNet score: {imagenet_score}"
                    )
                    scores.append(imagenet_score)

            if args.rank == 0:
                print(f"Shots {shot} Mean ImageNet score: {np.nanmean(scores)}")
                results["imagenet"].append(
                    {"shots": shot, "trials": scores, "mean": np.nanmean(scores)}
                )

    if args.rank == 0 and args.results_file is not None:
        with open(args.results_file, "w") as f:
            json.dump(results, f)


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


def get_query_set(train_dataset, query_set_size, seed):
    np.random.seed(seed)
    query_set = np.random.choice(len(train_dataset), len(train_dataset), replace=False)  #
    return [train_dataset[i] for i in query_set]


def prepare_eval_samples(test_dataset, num_samples, batch_size, seed):
    np.random.seed(seed)
    random_indices = np.random.choice(len(test_dataset), len(test_dataset), replace=False) #
    dataset = torch.utils.data.Subset(test_dataset, random_indices)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=custom_collate_fn,
    )
    return loader

def compute_effective_num_shots(num_shots, model_type):
    if model_type == "open_flamingo":
        return num_shots if num_shots > 0 else 2
    return num_shots


def custom_collate_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch


def evaluate_classification(
    args: argparse.Namespace,
    eval_model,
    seed: int = 42,
    num_shots: int = 8,
    no_kv_caching=False,
    dataset_name: str = "imagenet",
):
    """
    Evaluate a model on classification dataset.

    Args:
        eval_model (BaseEvalModel): model to evaluate
        imagenet_root (str): path to imagenet root for the specified split.
        seed (int, optional): random seed. Defaults to 42.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        dataset_name (str, optional): dataset name. Defaults to "imagenet".

    Returns:
        float: accuracy score
    """
    if args.model != "open_flamingo":
        raise NotImplementedError(
            "evaluate_classification is currently only supported for OpenFlamingo "
            "models"
        )
    batch_size = args.batch_size
    num_samples = args.num_samples
    model, tokenizer = eval_model.model, eval_model.tokenizer

    if dataset_name == "imagenet":
        train_dataset = ImageNetDataset(os.path.join(args.imagenet_root, "train"))
        test_dataset = ImageNetDataset(os.path.join(args.imagenet_root, "val"))
    elif dataset_name == "hateful_memes":
        train_dataset = HatefulMemesDataset(
            args.hateful_memes_image_dir_path,
            args.hateful_memes_train_annotations_json_path,
        )
        test_dataset = HatefulMemesDataset(
            args.hateful_memes_image_dir_path,
            args.hateful_memes_test_annotations_json_path,
        )
    else:
        raise ValueError(f"Unsupported dataset {dataset_name}")

    effective_num_shots = compute_effective_num_shots(num_shots, args.model)

    test_dataloader = prepare_eval_samples(
        test_dataset,
        args.num_samples if args.num_samples > 0 else len(test_dataset),
        batch_size,
        seed,
    )

    acc1 = 0
    acc5 = 0

    if dataset_name == "imagenet":
        prompt_text = "<image>Output:"
    elif dataset_name == "hateful_memes":
        prompt_text = "<image>is an image with: '{meme_text}' written on it. Is it hateful? Answer: "

    predictions = []

    np.random.seed(
        seed + args.rank
    )  # make sure each worker has a different seed for the random context samples
    for batch_idx, batch in tqdm(
        enumerate(test_dataloader),
        desc=f"Running inference {dataset_name}",
        disable=args.rank != 0,
    ):
        batch_images = []
        batch_text = []

        for idx in range(len(batch["image"])):
            # Choose a different set of random context samples for each sample
            # from the training set
            context_indices = np.random.choice(
                len(train_dataset), effective_num_shots, replace=False
            )

            in_context_samples = [train_dataset[i] for i in context_indices]

            if num_shots > 0:
                vision_x = [
                    eval_model.image_processor(data["image"]).unsqueeze(0)
                    for data in in_context_samples
                ]
            else:
                vision_x = []

            vision_x = vision_x + [
                eval_model.image_processor(batch["image"][idx]).unsqueeze(0)
            ]
            batch_images.append(torch.cat(vision_x, dim=0))

            def sample_to_prompt(sample):
                if dataset_name == "hateful_memes":
                    return prompt_text.replace("{meme_text}", sample["ocr"])
                else:
                    return prompt_text

            context_text = "".join(
                f"{sample_to_prompt(in_context_samples[i])}{in_context_samples[i]['class_name']}<|endofchunk|>"
                for i in range(effective_num_shots)
            )

            # Keep the text but remove the image tags for the zero-shot case
            if num_shots == 0:
                context_text = context_text.replace("<image>", "")

            batch_text.append(context_text)

        # shape [B, T_img, C, h, w]
        vision_x = torch.stack(batch_images, dim=0)
        # shape [B, T_img, 1, C, h, w] where 1 is the frame dimension
        vision_x = vision_x.unsqueeze(2)

        # Cache the context text: tokenize context and prompt,
        # e.g. '<context> a picture of a '
        text_x = [
            context_text + sample_to_prompt({k: batch[k][idx] for k in batch.keys()})
            for idx, context_text in enumerate(batch_text)
        ]

        ctx_and_prompt_tokenized = tokenizer(
            text_x,
            return_tensors="pt",
            padding="longest",
            max_length=2000,
        )

        ctx_and_prompt_input_ids = ctx_and_prompt_tokenized["input_ids"].to(
            eval_model.device
        )
        ctx_and_prompt_attention_mask = (
            ctx_and_prompt_tokenized["attention_mask"].to(eval_model.device).bool()
        )

        def _detach_pkvs(pkvs):
            """Detach a set of past key values."""
            return list([tuple([x.detach() for x in inner]) for inner in pkvs])

        if not no_kv_caching:
            eval_model.cache_media(
                input_ids=ctx_and_prompt_input_ids,
                vision_x=vision_x.to(eval_model.device),
            )

            with torch.no_grad():
                precomputed = eval_model.model(
                    vision_x=None,
                    lang_x=ctx_and_prompt_input_ids,
                    attention_mask=ctx_and_prompt_attention_mask,
                    clear_conditioned_layers=False,
                    use_cache=True,
                )

            precomputed_pkvs = _detach_pkvs(precomputed.past_key_values)
            precomputed_logits = precomputed.logits.detach()

        else:
            precomputed_pkvs = None
            precomputed_logits = None

        if dataset_name == "imagenet":
            all_class_names = IMAGENET_CLASSNAMES
        else:
            all_class_names = HM_CLASSNAMES

        if dataset_name == "imagenet":
            class_id_to_name = IMAGENET_1K_CLASS_ID_TO_LABEL
        else:
            class_id_to_name = HM_CLASS_ID_TO_LABEL

        overall_probs = []
        for class_name in all_class_names:
            past_key_values = None
            # Tokenize only the class name and iteratively decode the model's
            # predictions for this class.
            classname_tokens = tokenizer(
                class_name, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(eval_model.device)
            if classname_tokens.ndim == 1:  # Case: classname is only 1 token
                classname_tokens = torch.unsqueeze(classname_tokens, 1)

            classname_tokens = repeat(
                classname_tokens, "b s -> (repeat b) s", repeat=len(batch_text)
            )
            if not no_kv_caching:
                # Compute the outputs one token at a time, using cached
                # activations.

                # Initialize the elementwise predictions with the last set of
                # logits from precomputed; this will correspond to the predicted
                # probability of the first position/token in the imagenet
                # classname. We will append the logits for each token to this
                # list (each element has shape [B, 1, vocab_size]).
                elementwise_logits = [precomputed_logits[:, -2:-1, :]]

                for token_idx in range(classname_tokens.shape[1]):
                    _lang_x = classname_tokens[:, token_idx].reshape((-1, 1))
                    outputs = eval_model.get_logits(
                        lang_x=_lang_x,
                        past_key_values=(
                            past_key_values if token_idx > 0 else precomputed_pkvs
                        ),
                        clear_conditioned_layers=False,
                    )
                    past_key_values = _detach_pkvs(outputs.past_key_values)
                    elementwise_logits.append(outputs.logits.detach())

                # logits/probs has shape [B, classname_tokens + 1, vocab_size]
                logits = torch.concat(elementwise_logits, 1)
                probs = torch.softmax(logits, dim=-1)

                # collect the probability of the generated token -- probability
                # at index 0 corresponds to the token at index 1.
                probs = probs[:, :-1, :]  # shape [B, classname_tokens, vocab_size]

                gen_probs = (
                    torch.gather(probs, 2, classname_tokens[:, :, None])
                    .squeeze(-1)
                    .cpu()
                )

                class_prob = torch.prod(gen_probs, 1).numpy()
            else:
                # Compute the outputs without using cached
                # activations.

                # contatenate the class name tokens to the end of the context
                # tokens
                _lang_x = torch.cat([ctx_and_prompt_input_ids, classname_tokens], dim=1)
                _attention_mask = torch.cat(
                    [
                        ctx_and_prompt_attention_mask,
                        torch.ones_like(classname_tokens).bool(),
                    ],
                    dim=1,
                )

                outputs = eval_model.get_logits(
                    vision_x=vision_x.to(eval_model.device),
                    lang_x=_lang_x.to(eval_model.device),
                    attention_mask=_attention_mask.to(eval_model.device),
                    clear_conditioned_layers=True,
                )

                logits = outputs.logits.detach().float()
                probs = torch.softmax(logits, dim=-1)

                # get probability of the generated class name tokens
                gen_probs = probs[
                    :, ctx_and_prompt_input_ids.shape[1] - 1 : _lang_x.shape[1], :
                ]
                gen_probs = (
                    torch.gather(gen_probs, 2, classname_tokens[:, :, None])
                    .squeeze(-1)
                    .cpu()
                )
                class_prob = torch.prod(gen_probs, 1).numpy()

            overall_probs.append(class_prob)

        overall_probs = np.row_stack(overall_probs).T  # shape [B, num_classes]

        eval_model.uncache_media()

        def topk(probs_ary: np.ndarray, k: int) -> np.ndarray:
            """Return the indices of the top k elements in probs_ary."""
            return np.argsort(probs_ary)[::-1][:k]

        for i in range(len(batch_text)):
            highest_prob_idxs = topk(overall_probs[i], 5)

            top5 = [class_id_to_name[pred] for pred in highest_prob_idxs]

            y_i = batch["class_name"][i]
            acc5 += int(y_i in set(top5))
            acc1 += int(y_i == top5[0])

            predictions.append(
                {
                    "id": batch["id"][i],
                    "gt_label": y_i,
                    "pred_label": top5[0],
                    "pred_score": overall_probs[i][highest_prob_idxs[0]]
                    if dataset_name == "hateful_memes"
                    else None,  # only for hateful memes
                }
            )

    # all gather
    all_predictions = [None] * args.world_size
    torch.distributed.all_gather_object(all_predictions, predictions)  # list of lists
    if args.rank != 0:
        return

    all_predictions = [
        item for sublist in all_predictions for item in sublist
    ]  # flatten

    # Hack to remove samples with duplicate ids (only necessary for multi-GPU evaluation)
    all_predictions = {pred["id"]: pred for pred in all_predictions}.values()

    assert len(all_predictions) == len(test_dataset)  # sanity check

    if dataset_name == "hateful_memes":
        # return ROC-AUC score
        gts = [pred["gt_label"] for pred in all_predictions]
        pred_scores = [pred["pred_score"] for pred in all_predictions]
        return roc_auc_score(gts, pred_scores)
    else:
        # return top-1 accuracy
        acc1 = sum(
            int(pred["gt_label"] == pred["pred_label"]) for pred in all_predictions
        )
        return float(acc1) / len(all_predictions)


if __name__ == "__main__":
    main()
