from classification_utils import IMAGENET_1K_CLASS_ID_TO_LABEL
from torchvision.datasets import ImageFolder
from collections import defaultdict

class ImageNetDataset(ImageFolder):
    """Class to represent the ImageNet1k dataset."""

    def __init__(self, root, **kwargs):
        super().__init__(root=root, **kwargs)
        self.class_to_indices = self.build_class_index()

    def build_class_index(self):
        # 构建每个类别对应的索引列表
        class_to_indices = defaultdict(list)
        for idx, (_, class_id) in enumerate(self.imgs):
            class_to_indices[class_id].append(idx)
        return class_to_indices

    def __getitem__(self, idx):
        sample, target = super().__getitem__(idx)
        target_label = IMAGENET_1K_CLASS_ID_TO_LABEL[target]
        return {
            "id": idx,
            "image": sample,
            "class_id": target,  # numeric ID of the ImageNet class
            "class_name": target_label,  # human-readable name of ImageNet class
        }

    def get_data_list_by_class(self, class_id=None, class_name=None):
        # 根据 class_id 或 class_name 获取数据列表
        if class_id is not None:
            indices = self.class_to_indices[class_id]
        elif class_name is not None:
            class_id = {v: k for k, v in IMAGENET_1K_CLASS_ID_TO_LABEL.items()}.get(class_name)
            if class_id is None:
                return []  # 如果类名不存在
            indices = self.class_to_indices[class_id]
        else:
            return []

        # 返回指定类别的所有数据
        return [self.__getitem__(idx) for idx in indices]

