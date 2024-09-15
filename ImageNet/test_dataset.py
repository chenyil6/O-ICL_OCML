from imagenet_dataset import ImageNetDataset
from classification_utils import IMAGENET_CLASSNAMES_100
import os

support_set = []

train_dataset = ImageNetDataset(os.path.join("/data/hyh/imagenet/data", "train"))
# 取前100类对应的图片
for i in IMAGENET_CLASSNAMES_100:
    support_set.extend(train_dataset.get_data_list_by_class(class_name=i))

# 打印前10个数据以验证
print(support_set[:10])
