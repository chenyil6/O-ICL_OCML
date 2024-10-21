import torch
from classification_utils import IMAGENET_1K_CLASS_ID_TO_LABEL,IMAGENET_100_CLASS_ID_TO_LABEL
import math
import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)

def get_imagenet_prompt(label=None) -> str:
    return f"<image>Output:{label if label is not None else ''}{'<|endofchunk|>' if label is not None else ''}"

def get_topk_classifications(outputs,classnames_tokens,topk=2):
    overall_log_probs = torch.zeros(len(classnames_tokens))
    for idx,ct in enumerate(classnames_tokens):
        classname_tokens_num = len(ct)
        log_prob = 0
        valid = True  # 用于标记生成的token是否足够覆盖类别名称
        for i in range(classname_tokens_num):
            try:
                # Compute log probabilities instead of probabilities
                log_scores = torch.nn.functional.log_softmax(outputs.scores[i], dim=-1).squeeze().tolist()
                # Sum the log probabilities instead of multiplying probabilities
                log_prob += log_scores[ct[i]]
            except IndexError:
                log_prob = -float('inf')
                valid = False
                break  # 如果token数量不足，提前退出循环

        if valid:
            log_prob /= classname_tokens_num  # 归一化处理
            overall_log_probs[idx] = torch.exp(torch.tensor(log_prob))   # 将log概率转换为概率

    # 归一化类别概率
    overall_log_probs = overall_log_probs / overall_log_probs.sum()
    predicted_classnames, predicted_logprobs = get_predicted_classname(
        overall_log_probs,
        k=topk,
        class_id_to_name=IMAGENET_1K_CLASS_ID_TO_LABEL,
    )
    return predicted_classnames,predicted_logprobs,overall_log_probs


def get_topk_classifications_batch(outputs, classnames_tokens, topk,class_id_to_name = IMAGENET_1K_CLASS_ID_TO_LABEL):
    """
    Computes the top-k classifications for a batch of samples.

    Args:
        outputs: The outputs from model.generate, which includes scores.
        classnames_tokens: List of lists, tokenized class names.
        topk: int, number of top predictions to return.
        class_id_to_name: A dictionary mapping class indices to class names.

    Returns:
        predicted_classnames: List of lists containing predicted class names for each sample.
        predicted_logprobs: List of lists containing predicted log probabilities for each sample.
        overall_log_probs: Tensor of shape (batch_size, num_classes) containing the normalized probabilities.
    """
    batch_size = outputs.scores[0].size(0)  # Get the batch size from the first score tensor
    num_classes = len(classnames_tokens)
    num_generation_steps = len(outputs.scores)  # Number of tokens generated

    # Stack the scores into a tensor of shape (num_generation_steps, batch_size, vocab_size)
    scores_stack = torch.stack(outputs.scores, dim=0)  # shape (num_generation_steps, batch_size, vocab_size)

    # Initialize the overall log probabilities tensor
    overall_log_probs = torch.zeros(batch_size, num_classes)

    # Compute the log probabilities for each sample in the batch
    for batch_idx in range(batch_size):
        for class_idx in range(num_classes):
            ct = classnames_tokens[class_idx]
            classname_tokens_num = len(ct)
            log_prob = 0
            valid = True

            for i in range(classname_tokens_num):
                if i >= num_generation_steps:
                    # Generated tokens are less than class name tokens
                    log_prob = -float('inf')
                    valid = False
                    break

                # Get the log scores for this position
                logits = scores_stack[i, batch_idx, :]  # Shape: (vocab_size,)
                log_scores = torch.nn.functional.log_softmax(logits, dim=-1)

                # Get the log probability for the class token at position i
                token_id = ct[i]
                log_prob += log_scores[token_id].item()

            if valid:
                log_prob /= classname_tokens_num  # Normalize by the number of tokens
                overall_log_probs[batch_idx, class_idx] = torch.exp(torch.tensor(log_prob))

    # Normalize the probabilities for each sample
    overall_log_probs = overall_log_probs / overall_log_probs.sum(dim=1, keepdim=True)

    # Get top-k predictions for each sample
    predicted_classnames = []
    predicted_logprobs = []

    for batch_idx in range(batch_size):
        values, indices = torch.topk(overall_log_probs[batch_idx], k=topk)
        pred_classnames = [class_id_to_name[ix.item()] for ix in indices]
        pred_logprobs = values.tolist()
        predicted_classnames.append(pred_classnames)
        predicted_logprobs.append(pred_logprobs)

    return predicted_classnames, predicted_logprobs, overall_log_probs

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
