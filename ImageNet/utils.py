import torch
from classification_utils import IMAGENET_1K_CLASS_ID_TO_LABEL,IMAGENET_100_CLASS_ID_TO_LABEL
import math
import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)

def get_imagenet_prompt(label=None) -> str:
    return f"<image>Output:{label if label is not None else ''}{'<|endofchunk|>' if label is not None else ''}"

def get_topk_classifications(outputs, classnames_tokens, topk=2, temperature=2.0):
    overall_log_probs = torch.zeros(len(classnames_tokens))
    for idx, ct in enumerate(classnames_tokens):
        classname_tokens_num = len(ct)
        log_prob = 0
        valid = True  # 用于标记生成的token是否足够覆盖类别名称
        for i in range(classname_tokens_num):
            try:
                # Apply temperature scaling here
                # 在这里应用温度缩放，将 logits 除以 temperature
                scores = outputs.scores[i] / temperature
                # Compute log probabilities with temperature scaling
                log_scores = torch.nn.functional.log_softmax(scores, dim=-1).squeeze()
                # Sum the log probabilities instead of multiplying probabilities
                log_prob += log_scores[ct[i]]
            except IndexError:
                log_prob = -float('inf')
                valid = False
                break  # 如果token数量不足，提前退出循环

        if valid:
            # Normalize log_prob by classname_tokens_num to get average log_prob per token
            log_prob /= classname_tokens_num
            overall_log_probs[idx] = log_prob  # 不再进行取指数，因为接下来会使用 logsumexp

    # 使用logsumexp进行数值稳定的概率归一化
    log_probs_tensor = overall_log_probs
    # 减去logsumexp，使得所有log概率归一化
    log_probs_tensor = log_probs_tensor - torch.logsumexp(log_probs_tensor, dim=0)
    # 转换为概率
    probs_tensor = torch.exp(log_probs_tensor)

    predicted_classnames, predicted_probs = get_predicted_classname(
        probs_tensor,
        k=topk,
        class_id_to_name=IMAGENET_1K_CLASS_ID_TO_LABEL,
    )
    return predicted_classnames, predicted_probs, probs_tensor

def get_topk_classifications_batch(outputs, classnames_tokens, topk, temperature=2.0, class_id_to_name=IMAGENET_1K_CLASS_ID_TO_LABEL):
    """
    Computes the top-k classifications for a batch of samples.

    Args:
        outputs: The outputs from model.generate, which includes scores.
        classnames_tokens: List of lists, tokenized class names.
        topk: int, number of top predictions to return.
        temperature: float, temperature parameter for scaling logits.
        class_id_to_name: A dictionary mapping class indices to class names.

    Returns:
        predicted_classnames: List of lists containing predicted class names for each sample.
        predicted_probs: List of lists containing predicted probabilities for each sample.
        overall_probs: Tensor of shape (batch_size, num_classes) containing the normalized probabilities.
    """
    batch_size = outputs.scores[0].size(0)  # Get the batch size from the first score tensor
    num_classes = len(classnames_tokens)
    num_generation_steps = len(outputs.scores)  # Number of tokens generated

    # Stack the scores into a tensor of shape (num_generation_steps, batch_size, vocab_size)
    scores_stack = torch.stack(outputs.scores, dim=0)  # shape (num_generation_steps, batch_size, vocab_size)

    # Initialize the overall log probabilities tensor with -inf
    overall_log_probs = torch.full((batch_size, num_classes), -float('inf'))

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

                # Get the logits for this position and apply temperature scaling
                logits = scores_stack[i, batch_idx, :] / temperature  # Apply temperature scaling
                log_scores = torch.nn.functional.log_softmax(logits, dim=-1)

                # Get the log probability for the class token at position i
                token_id = ct[i]
                log_prob += log_scores[token_id]

            if valid:
                log_prob /= classname_tokens_num  # Normalize by the number of tokens
                overall_log_probs[batch_idx, class_idx] = log_prob  # Store the log probability

    # Use logsumexp to normalize the log probabilities for each sample
    # Subtract logsumexp from log probabilities to get normalized log probabilities
    log_probs_normalized = overall_log_probs - torch.logsumexp(overall_log_probs, dim=1, keepdim=True)

    # Convert to probabilities
    probs = torch.exp(log_probs_normalized)  # shape (batch_size, num_classes)

    # Get top-k predictions for each sample
    predicted_classnames = []
    predicted_probs = []

    for batch_idx in range(batch_size):
        values, indices = torch.topk(probs[batch_idx], k=topk)
        pred_classnames = [class_id_to_name[ix.item()] for ix in indices]
        pred_probs = values.tolist()
        predicted_classnames.append(pred_classnames)
        predicted_probs.append(pred_probs)

    return predicted_classnames, predicted_probs, probs

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