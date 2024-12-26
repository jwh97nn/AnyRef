import glob
import os
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset

from model.llava import conversation as conversation_lib
from model.llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    SEG_END_TOKEN,
    SEG_START_TOKEN,
)
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.data_processing import get_mask_from_json


class ReasonSeg(Dataset):
    def __init__(
        self,
        data_root: str = "/mnt/data/data/reason",
        all: bool = False,
        split: str = "train",
        only1: bool = False,
    ):
        self.data_root = data_root
        self.all = all
        self.split = split

        if split == "train":
            self.data_root = os.path.join(self.data_root, "train")
        elif split == "val":
            self.data_root = os.path.join(self.data_root, "val")
        else:
            raise ValueError("split must be train or val")

        self.json_path_list = sorted(glob.glob(self.data_root + "/*.json"))

        if split == "train":
            answer_file = os.path.join(data_root, "reason_answer_train.txt")
            with open(answer_file, "r") as f:
                answers = f.readlines()
                self.answers = [a.strip() for a in answers]
        else:
            self.answers = [""] * len(self.json_path_list)

        if only1:
            import random

            id = random.randint(0, len(self.json_path_list) - 1)
            # print(id)
            self.json_path_list = self.json_path_list[id : id + 1]
            self.answers = self.answers[id : id + 1]

    def __len__(self):
        return len(self.json_path_list)

    def __getitem__(self, index):
        json_path = self.json_path_list[index]
        image_path = json_path.replace(".json", ".jpg")

        image = cv2.imread(image_path)
        h, w, _ = image.shape
        mask, sents, is_sentences = get_mask_from_json(json_path, image)
        if len(sents) > 1:
            sent = sents[np.random.randint(len(sents))]
        else:
            sent = sents[0]
        mask = (mask == 1).astype(np.uint8)

        if self.split == "train":
            answer = self.answers[index]
        else:  # currently no answer for "val"
            answer = self.answers[0]

        return dict(
            file_name=image_path,
            height=h,
            width=w,
            gt_classes=[sent],
            # gt_classes=sents,
            gt_masks=torch.from_numpy(mask).unsqueeze(0),
            answer=answer,
            is_sentences=is_sentences,
        )


class ReasonSegTokenized(ReasonSeg):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_root: str = "/mnt/data/data/reason",
        # dataset: str = "refcoco", #,refcoco+,refcocog", #,refclef",
        # use2017: bool = True,
        clip_version: str = "openai/clip-vit-large-patch14/",
        split: str = "train",  # "testA", "testB"
        image_size: int = 1024,  # for SAM
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        all: bool = False,
        conv_type: str = "llava_v1",
        num_obj_token: int = 1,
        with_bbox: bool = False,
        clip_resize_wo_crop: bool = True,
        sampled_class_num: int = 1,
        itisseg: bool = False,
        seg_start_end: bool = False,
        only1: bool = False,
    ):
        super().__init__(data_root=data_root, all=all, split=split, only1=only1)

        self.tokenizer = tokenizer
        self.obj_token = "[SEG]"
        self.clip_processor = transformers.CLIPImageProcessor.from_pretrained(
            clip_version
        )

        self.sam_image_size = image_size
        self.transforms = ResizeLongestSide(image_size)
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)

        self.all = all  # whether to use all classes

        self.conv_type = conv_type

        self.num_obj_token = num_obj_token
        if num_obj_token > 1:
            self.num_obj_token = num_obj_token
            self.obj_token = [f"[SEG{i}]" for i in range(num_obj_token)]

        self.with_bbox = with_bbox
        if self.with_bbox:
            self.loc_token = [f"[LOC{i}]" for i in range(101)]

        self.clip_resize_wo_crop = clip_resize_wo_crop
        if self.clip_resize_wo_crop:
            self.clip_processor.do_center_crop = False

        self.sampled_class_num = sampled_class_num

        self.itisseg = itisseg
        self.seg_start_end = seg_start_end

    def sam_preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.sam_image_size - h
        padw = self.sam_image_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, index):
        dataset_dict = super().__getitem__(index)

        image = cv2.imread(dataset_dict["file_name"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        clip_image = self.clip_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]
        if self.clip_resize_wo_crop:
            clip_image = F.interpolate(
                clip_image.unsqueeze(0),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )[0]
        sam_image = self.transforms.apply_image(image)

        sam_resized_size = sam_image.shape[:2]
        sam_image = self.sam_preprocess(
            torch.from_numpy(sam_image).permute(2, 0, 1).contiguous()
        )

        gt_classes = dataset_dict["gt_classes"]
        gt_answer = dataset_dict["answer"]
        gt_masks = dataset_dict["gt_masks"]
        is_sentences = dataset_dict["is_sentences"]

        class_text = gt_classes[0]
        if self.seg_start_end:
            class_text = f"{SEG_START_TOKEN}{class_text}{SEG_END_TOKEN}"

        if is_sentences:
            # question = f"{DEFAULT_IMAGE_TOKEN}\nCan you segment '{class_text}' in this image?"
            question = (
                f"{DEFAULT_IMAGE_TOKEN}\n{class_text} Can you segment it in this image?"
            )
        else:
            question = (
                f"{DEFAULT_IMAGE_TOKEN}\nCan you segment {class_text} in this image?"
            )

        class_obj_text = f"{gt_answer}{self.obj_token}"
        if gt_answer != "":
            answer = f"{class_obj_text}."
        else:
            answer = f"it is {self.obj_token}."
        if self.itisseg:
            answer = f"it is {self.obj_token}."

        conv = conversation_lib.conv_templates[self.conv_type].copy()

        conv.messages = []
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], answer)
        conversation = conv.get_prompt()
        # print(conversation)

        dataset_dict["clip_image"] = clip_image
        dataset_dict["sam_image"] = sam_image
        dataset_dict["sam_resized_size"] = sam_resized_size

        dataset_dict["question"] = question
        dataset_dict["conversation"] = conversation

        dataset_dict["gt_classes"] = gt_classes
        dataset_dict["gt_masks"] = gt_masks
        dataset_dict["gt_bbox"] = None

        return dataset_dict
