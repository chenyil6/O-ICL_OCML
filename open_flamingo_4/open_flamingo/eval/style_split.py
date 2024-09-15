import os
import numpy as np


class COCOStyleDataset():
    def __init__(self):
        self.val_data_dir = "/data/wyl/coco_data/val2014"

        self.train_name = []
        with open('/data/ll/StyleCaption/data/senticap_dataset/train_neg.txt', 'r', encoding='utf-8') as f:
            self.train_name = [i.split(",")[0].strip("\n") for i in f.readlines()[1:]]
        print(len(self.train_name))

        self.train_data = {}
        for tn in self.train_name:
            self.train_data[self.name2id(tn)] = os.path.join(self.val_data_dir, tn)

        self.test_name = []
        with open('/data/ll/StyleCaption/data/senticap_dataset/test_neg.txt', 'r', encoding='utf-8') as f:
            self.test_name = [i.split(",")[0].strip("\n") for i in f.readlines()[1:]]
        print(len(self.test_name))

        self.test_data = {}
        for tn in self.test_name:
            self.test_data[self.name2id(tn)] = os.path.join(self.val_data_dir, tn)

        self.val_name = []
        with open('/data/ll/StyleCaption/data/senticap_dataset/val_neg.txt', 'r', encoding='utf-8') as f:
            self.val_name = [i.split(",")[0].strip("\n") for i in f.readlines()[1:]]
        print(len(self.val_name))

        self.val_data = {}
        for tn in self.val_name:
            self.val_data[self.name2id(tn)] = os.path.join(self.val_data_dir, tn)

    def name2id(self, name):
        return int(name.split("_")[2].split(".")[0])

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def get_val_data(self):
        return self.val_data


class FlickrStyleDataset():
    def __init__(self):
        self.val_data_dir = "/data/pyz/data/flickr8k/Flickr8k_Dataset/"

        # since the flickrstyle dataset only has train set open-source,
        # we divide the set as the rate of 7:3 by ourselves
        whole_dataset = np.load('/data/ll/StyleCaption/data/FlickrStyle/train_split.npy')
        whole_num = len(whole_dataset)
        train_num = 0.7 * whole_num
        # test_num = 0.3 * whole_num

        self.train_name = []
        self.train_data = []
        self.test_name = []
        self.test_data = []

        for idx, sample in enumerate(whole_dataset):
            if idx < train_num:
                self.train_name.append(sample.split(".")[0])
                self.train_data.append(os.path.join(self.val_data_dir, sample))
            else:
                self.test_name.append(sample.split(".")[0])
                self.test_data.append(os.path.join(self.val_data_dir, sample))

    def get_train_data(self):
        return self.train_data, self.train_name

    def get_test_data(self):
        return self.test_data, self.test_name


class COCOVqaDataset():
    def __init__(self):
        self.train_data_dir = "/data/wyl/coco_data/train2014"
        self.val_data_dir = "/data/wyl/coco_data/val2014"

        self.train_id = []
        with open('/data/wyl/arctic-captions/splits/coco_train.txt', 'r', encoding='utf-8') as f:
            self.train_name = [i.strip("\n") for i in f.readlines()]
        print(len(self.train_name))

        self.train_data = {}
        for tn in self.train_name:
            self.train_data[self.name2id(tn)] = os.path.join(self.train_data_dir, tn) if tn.find(
                "train") != -1 else os.path.join(self.val_data_dir, tn)

        self.test_name = []
        with open("/data/wyl/arctic-captions/splits/coco_val.txt", 'r', encoding='utf-8') as f:
            self.test_name = [i.strip("\n") for i in f.readlines()]
        with open('/data/wyl/arctic-captions/splits/coco_restval.txt', 'r', encoding='utf-8') as f:
            self.test_name.extend([i.strip("\n") for i in f.readlines()])
        print(len(self.test_name))

        self.test_data = {}
        for tn in self.test_name:
            self.test_data[self.name2id(tn)] = os.path.join(self.val_data_dir, tn)

        # self.val_name = []
        # with open('/data/wyl/arctic-captions/splits/coco_val.txt', 'r', encoding='utf-8') as f:
        #     self.val_name = [i.strip("\n") for i in f.readlines()]
        # print(len(self.val_name))

        # self.val_data = {}
        # for tn in self.val_name:
        #     self.val_data[self.name2id(tn)] = os.path.join(self.val_data_dir, tn)

    def name2id(self, name):
        return int(name.split("_")[2].split(".")[0])

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def get_val_data(self):
        return self.val_data