import os
import random
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from PIL import Image
from torch.utils.data import Dataset

from model.llava import conversation as conversation_lib
from model.llava.constants import DEFAULT_IMAGE_TOKEN
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.ade_ins_category import ADE_INS_CATEGORY


class ADEIns(Dataset):
    def __init__(
        self,
        is_train: bool = True,
        root_dir: str = "/mnt/data/data/ADEChallengeData2016/",
        filter_area: float = None,
    ):
        if is_train:
            sub_dir = "training"
        else:
            sub_dir = "validation"
        image_ids = os.listdir(os.path.join(root_dir, "images", sub_dir))

        self.images = []
        self.labels = []
        for image_id in image_ids:
            image_path = os.path.join(root_dir, "images", sub_dir, image_id)
            label_path = os.path.join(
                root_dir,
                "annotations_instance",
                sub_dir,
                image_id.replace("jpg", "png"),
            )
            self.images.append(image_path)
            self.labels.append(label_path)

        self.filter_area = filter_area

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_file = self.images[index]
        label_file = self.labels[index]

        ins_seg = np.asarray(Image.open(label_file))
        instance_cat_ids = ins_seg[..., 0]
        instance_ins_ids = ins_seg[..., 1]
        height, width = instance_ins_ids.shape

        bbox, gt_classes, gt_masks = [], [], []
        for thing_id in np.unique(instance_ins_ids):
            if thing_id == 0:
                continue

            mask = instance_ins_ids == thing_id
            if self.filter_area is not None:
                if mask.sum() / (mask.shape[0] * mask.shape[1]) < self.filter_area:
                    continue
            gt_masks.append(torch.as_tensor(mask, dtype=torch.uint8))

            instance_cat_id = np.unique(instance_cat_ids[mask])
            gt_classes.append(int(instance_cat_id))

            inds = np.nonzero(mask)
            ymin, ymax = inds[0].min(), inds[0].max()
            xmin, xmax = inds[1].min(), inds[1].max()
            bbox.append([int(xmin), int(ymin), int(xmax), int(ymax)])

        if len(gt_masks) == 0:
            return None

        return dict(
            file_name=image_file,
            height=height,
            width=width,
            bbox=torch.tensor(bbox, dtype=torch.float32),
            gt_classes=torch.tensor(gt_classes),
            gt_masks=torch.stack(gt_masks, dim=0),
        )


class ADEInstanceTokenized(ADEIns):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        clip_version: str = "openai/clip-vit-large-patch14/",
        is_train: bool = True,
        root_dir: str = "/mnt/data/data/ADEChallengeData2016/",
        image_size: int = 1024,  # for SAM
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        filter_area: float = None,
        all: bool = False,
        conv_type: str = "llava_v1",
        with_bbox: bool = False,
        clip_resize_wo_crop: bool = False,
        sampled_class_num: int = 1,
        overfit: bool = False,
    ):
        super().__init__(is_train=is_train, root_dir=root_dir, filter_area=filter_area)
        self.is_train = is_train
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

        self.with_bbox = with_bbox
        if self.with_bbox:
            self.loc_token = [f"[LOC{i}]" for i in range(101)]

        self.clip_resize_wo_crop = clip_resize_wo_crop
        if self.clip_resize_wo_crop:
            self.clip_processor.do_center_crop = False

        self.sampled_class_num = sampled_class_num

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
        # print(dataset_dict.keys())

        if dataset_dict is None:
            return self.__getitem__(random.randint(0, self.__len__() - 1))

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
        # print(gt_classes)
        gt_masks = dataset_dict["gt_masks"]

        orig_gt_bbox = dataset_dict["bbox"]  # [N, 4], [x1, y1, x2, y2] [x1,y1]=top-left
        # convert xywh to xyxy
        # orig_gt_bbox[:, 2:] += orig_gt_bbox[:, :2]    # [x1, y1, x2, y2]  top-left, bottom-right
        gt_bbox = orig_gt_bbox.clone()
        # normalize to [0, 1]
        gt_bbox[:, 0::2] /= dataset_dict["width"]
        gt_bbox[:, 1::2] /= dataset_dict["height"]
        # keep 2 decimals
        gt_bbox = torch.round(gt_bbox * 100).long()

        unique_classes = set(gt_classes.tolist())

        unexist = False
        if random.random() < 0.0:
            # Tag: randomly sample a class that does not exist in the image
            unexist = True
            sampled_class_num = 1
            unexist_classes = [
                k for k in ADE_INS_CATEGORY.keys() if k not in unique_classes
            ]
            sampled_classes = random.sample(unexist_classes, sampled_class_num)
        else:
            # Tag: normal sampling
            sampled_class_num = random.randint(1, self.sampled_class_num)
            if sampled_class_num > len(unique_classes):
                sampled_class_num = len(unique_classes)
            # sampled_class_num = 1   # Tag: only select 1 class per image
            sampled_classes = random.sample(unique_classes, sampled_class_num)

        if self.all:
            sampled_class_num = len(unique_classes)
            sampled_classes = list(unique_classes)

        gt_classes_text, gt_classes_text_obj = [], []
        sampled_class, sampled_mask, sampled_bbox = [], [], []
        for c in sampled_classes:
            sampled_idx = gt_classes == c

            sampled_class.append(gt_classes[sampled_idx])
            sampled_mask.append(gt_masks[sampled_idx])
            sampled_bbox.append(orig_gt_bbox[sampled_idx])
            sampled_bbox_ = gt_bbox[sampled_idx]

            class_text = ADE_INS_CATEGORY[c]
            gt_classes_text.append(class_text)

            instance_num = int(sampled_idx.sum())
            if self.with_bbox:  # person [x1][y1][x2][y2][seg], person...
                text = []
                for i in range(instance_num):
                    bbox = sampled_bbox_[i]
                    loc_tokens = [f"[LOC{int(j)}]" for j in bbox]
                    # print(loc_tokens)
                    text.append(f"{class_text}{''.join(loc_tokens)}{self.obj_token}")
                gt_classes_text_obj.append(",".join(text))
            else:
                gt_classes_text_obj.append(
                    f"{class_text}{self.obj_token * instance_num}"
                )

        gt_classes = torch.cat(sampled_class, dim=0)
        gt_masks = torch.cat(sampled_mask, dim=0)
        orig_gt_bbox = torch.cat(sampled_bbox, dim=0)

        if sampled_class_num == 1:
            class_text = gt_classes_text[0]
            class_obj_text = gt_classes_text_obj[0]
        else:
            class_text = ", ".join(gt_classes_text[:-1]) + f" and {gt_classes_text[-1]}"
            class_obj_text = (
                ", ".join(gt_classes_text_obj[:-1]) + f" and {gt_classes_text_obj[-1]}"
            )
        # print(class_obj_text)

        question = f"{DEFAULT_IMAGE_TOKEN}\nCan you segment {class_text} in this image?"
        if not unexist:
            # target = f"{class_obj_text}.</s>"
            answer = f"{class_obj_text}."
        else:
            answer = f"there is no {class_text} in this image."

        # conv = conversation_lib.default_conversation.copy()
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
        dataset_dict["gt_bbox"] = orig_gt_bbox

        return dataset_dict
