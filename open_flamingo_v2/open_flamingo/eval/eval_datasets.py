import json
import os
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import numpy as np

from open_flamingo.eval.classification_utils import IMAGENET_1K_CLASS_ID_TO_LABEL


class VQADataset(Dataset):
    def __init__(
        self, train_image_dir_path, question_path, annotations_path, is_train, dataset_name,val_image_dir_path
    ):
        self.dataset_name = dataset_name
        if self.dataset_name == "vqa_cp":
            self.questions = json.load(open(question_path, "r"))
            self.answers = json.load(open(annotations_path, "r"))
        else:
            self.questions = json.load(open(question_path, "r"))["questions"]
            self.answers = json.load(open(annotations_path, "r"))["annotations"]

        self.train_image_dir_path = train_image_dir_path
        self.is_train = is_train
        if self.dataset_name in {"vqav2", "ok_vqa","vqa_cp"}:
            self.img_coco_split = self.train_image_dir_path.strip("/").split("/")[-1]
            assert self.img_coco_split in {"train2014", "val2014", "test2015"}
        if val_image_dir_path is not None:
            self.val_image_dir_path = val_image_dir_path
            self.img_coco_split2 = self.val_image_dir_path.strip("/").split("/")[-1]
            assert self.img_coco_split2 in {"val2014"}
        if not self.is_train:
            if self.dataset_name == "vizwiz":
                siir_retrieval_path = "/data/ll/StyleCaption/style_clip/vqa/vizwiz_validation_SIIR.npy"
                sqqr_retrieval_path = "/data/ll/StyleCaption/style_clip/vqa/vizwiz_validation_SQQR.npy"
                sqaqar_retrieval_path = "/data/ll/StyleCaption/style_clip/vqa/vizwiz_validation_SQAQAR.npy"
                sqaqar_siir_retrieval_path = "/data/ll/StyleCaption/style_clip/vqa/vizwiz_validation_SQAQAR_generated_SIIR.npy"
                sqaqar_rs_retrieval_path = "/data/ll/StyleCaption/style_clip/vqa/vizwiz_validation_SQAQAR_generated_RS.npy"
                sqaqar_siir_4shot_retrieval_path = "/data/ll/StyleCaption/style_clip/vqa/vizwiz_validation_SQAQAR_generated_SIIR_4shot.npy"
                sqaqar_rs_4shot_retrieval_path = "/data/ll/StyleCaption/style_clip/vqa/vizwiz_validation_SQAQAR_generated_RS_4shot.npy"

                self.siir_set = np.load(siir_retrieval_path, allow_pickle=True).item()
                self.sqqr_set = np.load(sqqr_retrieval_path, allow_pickle=True).item()
                self.sqaqar_set = np.load(sqaqar_retrieval_path, allow_pickle=True).item()
                self.sqaqar_siir_set = np.load(sqaqar_siir_retrieval_path, allow_pickle=True).item()
                self.sqaqar_rs_set = np.load(sqaqar_rs_retrieval_path, allow_pickle=True).item()
                self.sqaqar_siir_4shot_set = np.load(sqaqar_siir_4shot_retrieval_path, allow_pickle=True).item()
                self.sqaqar_rs_4shot_set = np.load(sqaqar_rs_4shot_retrieval_path, allow_pickle=True).item()
            elif self.dataset_name == "ok_vqa":
                siir_retrieval_path = "/data/ll/StyleCaption/style_clip/vqa/okvqa_validation_SIIR.npy"
                sqqr_retrieval_path = "/data/ll/StyleCaption/style_clip/vqa/okvqa_validation_SQQR.npy"
                sqaqar_retrieval_path = "/data/ll/StyleCaption/style_clip/vqa/okvqa_validation_SQAQAR.npy"
                sqaqar_siir_retrieval_path = "/data/ll/StyleCaption/style_clip/vqa/okvqa_validation_SQAQAR_generated_siir.npy"
                sqaqar_rs_retrieval_path = "/data/ll/StyleCaption/style_clip/vqa/okvqa_validation_SQAQAR_generated_RS.npy"

                self.siir_set = np.load(siir_retrieval_path, allow_pickle=True).item()
                self.sqqr_set = np.load(sqqr_retrieval_path, allow_pickle=True).item()
                self.sqaqar_set = np.load(sqaqar_retrieval_path, allow_pickle=True).item()
                self.sqaqar_siir_set = np.load(sqaqar_siir_retrieval_path, allow_pickle=True).item()
                self.sqaqar_rs_set = np.load(sqaqar_rs_retrieval_path, allow_pickle=True).item()
            elif self.dataset_name == "vqa_cp":
                retrieval_path = "/data/chy/openflamingo_v2/clip_files/validation_VQAcp4.npy"
                self.retrieval_set = np.load(retrieval_path, allow_pickle=True).item()


    def __len__(self):
        return len(self.questions)

    def id2item(self, idx):
        question = self.questions[idx]
        answers = self.answers[idx]
        img_path = self.get_img_path(question)
        image = Image.open(img_path)
        mismatch_question = random.choice(self.questions)
        return {
            "image": image,
            "image_id": question['image_id'],
            "question": question["question"],
            "answers": [a["answer"] for a in answers["answers"]],
            "question_id": question["question_id"],
            "QA_question": mismatch_question["question"],
        }
    def id2idx(self,id):
        if self.dataset_name in {"vqav2", "ok_vqa"}:
            train_path ="/data/pyz/data/vqav2/v2_OpenEnded_mscoco_train2014_questions.json"
            train_set = json.load(open(train_path, "r"))["questions"]
        elif self.dataset_name == "vqa_cp":
            train_path = "/data/share/chy/VQA_CP/vqacp_v2_train_questions.json"
            train_set = json.load(open(train_path, "r"))
        for idx, question in enumerate(train_set):
            if question['image_id'] == id:
                return idx

    def get_img_path(self, question):
        if self.dataset_name in {"vqav2", "ok_vqa"}:
            return os.path.join(
                self.train_image_dir_path,
                f"COCO_{self.img_coco_split}_{question['image_id']:012d}.jpg"
                if self.is_train
                else f"COCO_{self.img_coco_split}_{question['image_id']:012d}.jpg",
            )
        elif self.dataset_name=="vqa_cp":
            if question['coco_split'] == 'train2014':
                image_path = os.path.join(self.train_image_dir_path,
                                          f"COCO_{question['coco_split']}_{question['image_id']:012d}.jpg")
            elif question['coco_split'] == 'val2014':
                image_path = os.path.join(self.val_image_dir_path,
                                          f"COCO_{question['coco_split']}_{question['image_id']:012d}.jpg")
            return image_path
        elif self.dataset_name == "vizwiz":
            return os.path.join(self.train_image_dir_path, question["image_id"])
        elif self.dataset_name == "textvqa":
            return os.path.join(self.train_image_dir_path, f"{question['image_id']}.jpg")
        else:
            raise Exception(f"Unknown VQA dataset {self.dataset_name}")

    def __getitem__(self, idx):
        question = self.questions[idx]
        img_path = self.get_img_path(question)
        image = Image.open(img_path)
        image.load()
        mismatch_question = random.choice(self.questions)
        results = {
            "image": image,
            "image_id": question['image_id'],
            "question": question["question"],
            "question_id": question["question_id"],
            "QA_question": mismatch_question["question"],
        }
        if self.answers is not None:
            answers = self.answers[idx]
            results["answers"] = [a["answer"] for a in answers["answers"]]
            results["question_type"] = answers["question_type"]


        if not self.is_train:
            question_id = question['question_id']
            if self.dataset_name == "vizwiz":
                results["SIIR"] = [i[2] for i in self.siir_set[question_id]["SIIR"]]
                results["SIIR_0"] = [i[2] for i in self.siir_set[question_id]["SIIR_0"]]
                results["SIIR_1"] = [i[2] for i in self.siir_set[question_id]["SIIR_1"]]
                results["SIIR_2"] = [i[2] for i in self.siir_set[question_id]["SIIR_2"]]
                results["SQQR"] = [i[2] for i in self.sqqr_set[question_id]["SQQR"]]
                results["SQAQAR"] = [i[2] for i in self.sqaqar_set[question_id]["SQAQAR"]]
                # 32 shot
                results["SQAQAR_SIIR"] = [i[2] for i in self.sqaqar_siir_set[question_id]["SQAQAR_SIIR"]]
                results["SQAQAR_RS"] = [i[2] for i in self.sqaqar_rs_set[question_id]["SQAQAR_rs"]]
                # 4 shot
                results["SQAQAR_SIIR_4shot"] = [i[2] for i in self.sqaqar_siir_4shot_set[question_id]["SQAQAR_SIIR_4"]]
                results["SQAQAR_RS_4shot"] = [i[2] for i in self.sqaqar_rs_4shot_set[question_id]["SQAQAR_RS_4"]]
            elif self.dataset_name == "ok_vqa":
                results["SIIR"] = [i[2] for i in self.siir_set[question_id]["SIIR"]]
                results["SIIR_0"] = [i[2] for i in self.siir_set[question_id]["SIIR_0"]]
                results["SIIR_1"] = [i[2] for i in self.siir_set[question_id]["SIIR_1"]]
                results["SIIR_2"] = [i[2] for i in self.siir_set[question_id]["SIIR_2"]]
                results["SQQR"] = [i[2] for i in self.sqqr_set[question_id]["SQQR"]]
                results["SQAQAR"] = [i[2] for i in self.sqaqar_set[question_id]["SQAQAR"]]
                # 4 shot
                results["SQAQAR_SIIR"] = [i[2] for i in self.sqaqar_siir_set[question_id]["SQAQAR_SIIR"]]
                results["SQAQAR_RS"] = [i[2] for i in self.sqaqar_rs_set[question_id]["SQAQAR_RS"]]
            elif self.dataset_name == "vqa_cp":
                question_id = question['question_id']
                results["SI"] = [i[2] for i in self.retrieval_set[question_id]["SI"]]
                results["SI_Q"] = [i[2] for i in self.retrieval_set[question_id]["SI_Q"]]
                results["SQ"] = [i[2] for i in self.retrieval_set[question_id]["SQ"]]
                # results["SQ_I"] = [i[2] for i in self.retrieval_set[question_id]["SQ_I"]]
                # results["SI_1"] = [i[2] for i in self.retrieval_set[question_id]["SI_1"]]
                # results["SI_2"] = [i[2] for i in self.retrieval_set[question_id]["SI_2"]]
        return results

class ImageNetDataset(ImageFolder):
    """Class to represent the ImageNet1k dataset."""

    def __init__(self, root, **kwargs):
        super().__init__(root=root, **kwargs)

    def __getitem__(self, idx):
        sample, target = super().__getitem__(idx)
        target_label = IMAGENET_1K_CLASS_ID_TO_LABEL[target]
        return {
            "id": idx,
            "image": sample,
            "class_id": target,  # numeric ID of the ImageNet class
            "class_name": target_label,  # human-readable name of ImageNet class
        }


class HatefulMemesDataset(Dataset):
    def __init__(self, image_dir_path, annotations_path):
        self.image_dir_path = image_dir_path
        with open(annotations_path, "r") as f:
            self.annotations = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.image_dir_path, annotation["img"].split("/")[-1])
        image = Image.open(img_path)
        image.load()
        return {
            "id": annotation["id"],
            "image": image,
            "ocr": annotation["text"],
            "class_name": "yes" if annotation["label"] == 1 else "no",
            "class_id": annotation["label"],
        }


