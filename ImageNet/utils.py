import torch
from classification_utils import IMAGENET_1K_CLASS_ID_TO_LABEL,IMAGENET_CLASSNAMES
import math
import numpy as np


def get_imagenet_prompt(label=None) -> str:
    return f"<image>Output:{label if label is not None else ''}{'<|endofchunk|>' if label is not None else ''}"

def get_topk_classifications(outputs,classnames_tokens,topk):
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

def custom_collate_fn(batch):
    """
    Collate function for DataLoader that collates a list of dicts into a dict of lists.
    """
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch

def prepare_eval_samples(test_dataset,batch_size, seed):
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
