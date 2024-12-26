import os
import random
from typing import List

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from PIL import Image
from torch.utils.data import Dataset

from model.llava import conversation as conversation_lib
from model.llava.constants import (
    AUDIO_REF_END_TOKEN,
    AUDIO_REF_START_TOKEN,
    AUDIO_REF_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    SEG_END_TOKEN,
    SEG_START_TOKEN,
)
from model.segment_anything.utils.transforms import ResizeLongestSide

try:
    from model.ImageBind.data import load_and_transform_audio_data
except:
    print("torchaudio not installed")
    raise NotImplementedError


avs_category_to_class = {
    "helicopter": "helicopter",
    "mynah_bird_singing": "bird",
    "typing_on_computer_keyboard": "keyboard",
    "playing_violin": "violin",
    "playing_glockenspiel": "glockenspiel",
    "playing_piano": "piano",
    "lions_roaring": "lion",
    "baby_laughter": "baby",
    "male_speech": "male",
    "lawn_mowing": "lawn mower",
    "playing_ukulele": "ukulele",
    "playing_tabla": "tabla",
    "driving_buses": "bus",
    "cap_gun_shooting": "cap gun",
    "chainsawing_trees": "chainsaw",
    "playing_acoustic_guitar": "guitar",
    "cat_meowing": "cat",
    "female_singing": "female",
    "ambulance_siren": "ambulance",
    "dog_barking": "dog",
    "horse_clip-clop": "horse",
    "coyote_howling": "coyote",
    "race_car": "car",
}


class AVSObject(Dataset):
    def __init__(
        self,
        # is_train: bool = True,
        split: str = "train",
        root_dir: str = "/mnt/data/data/avsbench/Single-source/",
        csv_file: str = "s4_meta_data.csv",
        image_subdir: str = "s4_data/visual_frames/",
        audio_subdir: str = "s4_data/audio_wav/",
        mask_subdir: str = "s4_data/gt_masks/",
        convert_classname: bool = True,
        original_resolution: bool = True,
    ):
        self.is_train = split == "train"
        self.root_dir = root_dir
        self.image_subdir = image_subdir
        self.audio_subdir = audio_subdir
        self.mask_subdir = mask_subdir
        self.convert_classname = convert_classname

        if original_resolution:
            self.image_subdir = image_subdir.replace(
                "visual_frames", "visual_frames_original_resolution"
            )
            # self.mask_subdir = mask_subdir.replace("gt_masks", "gt_masks_original_resolution")

        df = pd.read_csv(os.path.join(root_dir, csv_file), sep=",")
        self.split = split
        self.df = df[df["split"] == self.split]

        print(f"avs_object {split} has {self.__len__()} samples.")

    def __len__(self):
        if self.split == "train":
            return len(self.df)
        elif self.split == "val":
            return len(self.df) * 5
        elif self.split == "test":
            return len(self.df) * 5
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        if self.split == "train":
            data = self.df.iloc[index]
        elif self.split == "val":
            data = self.df.iloc[index // 5]
        elif self.split == "test":
            data = self.df.iloc[index // 5]

        video_name = data[0]
        category = data[2]

        image_dir = os.path.join(
            self.root_dir, self.image_subdir, self.split, category, video_name
        )
        audio_dir = os.path.join(self.root_dir, self.audio_subdir, self.split, category)
        mask_dir = os.path.join(
            self.root_dir, self.mask_subdir, self.split, category, video_name
        )

        if self.split == "train":
            id = 1
        elif self.split == "val":
            id = index % 5 + 1
        elif self.split == "test":
            id = index % 5 + 1
        else:
            raise NotImplementedError

        # for id in ids:
        image_file = os.path.join(image_dir, f"{video_name}_{id}.png")
        assert os.path.exists(image_file), f"{image_file} does not exist"
        audio_file = os.path.join(audio_dir, f"{video_name}.wav")
        mask_file = os.path.join(mask_dir, f"{video_name}_{id}.png")

        gt_mask = Image.open(mask_file).convert(mode="1")
        gt_mask = np.array(gt_mask, dtype=np.uint8)
        # h, w = gt_mask.shape
        gt_mask = torch.as_tensor(gt_mask, dtype=torch.uint8).unsqueeze(0)

        if self.convert_classname:
            category = avs_category_to_class[category]

        return dict(
            file_name=image_file,
            audio_file_name=audio_file,
            height=None,
            width=None,
            bbox=None,
            # gt_classes=torch.tensor(gt_classes),
            gt_classes=[category],
            # gt_masks=torch.stack(gt_mask, dim=0),
            gt_masks=gt_mask,  # [1, h, w]
        )


class AVSObjectTokenized(AVSObject):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        root_dir: str = "/mnt/data/data/avsbench/Single-source/",
        csv_file: str = "s4_meta_data.csv",
        image_subdir: str = "s4_data/visual_frames/",
        audio_subdir: str = "s4_data/audio_wav/",
        mask_subdir: str = "s4_data/gt_masks/",
        clip_version: str = "openai/clip-vit-large-patch14/",
        split: str = "train",
        image_size: int = 1024,  # for SAM
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        conv_type: str = "llava_v1",
        clip_resize_wo_crop: bool = True,
        convert_classname: bool = True,
        original_resolution: bool = True,
        itisseg: bool = False,
        multi_modality: bool = False,
        seg_start_end: bool = False,
        placehold: bool = False,
    ):
        super().__init__(
            split=split,
            root_dir=root_dir,
            csv_file=csv_file,
            image_subdir=image_subdir,
            audio_subdir=audio_subdir,
            mask_subdir=mask_subdir,
            convert_classname=convert_classname,
            original_resolution=original_resolution,
        )
        self.is_train = split == "train"
        self.tokenizer = tokenizer
        self.obj_token = "[SEG]"
        self.clip_processor = transformers.CLIPImageProcessor.from_pretrained(
            clip_version
        )

        self.conv_type = conv_type
        self.sam_image_size = image_size
        self.transforms = ResizeLongestSide(image_size)
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)

        self.clip_resize_wo_crop = clip_resize_wo_crop
        if self.clip_resize_wo_crop:
            self.clip_processor.do_center_crop = False

        self.itisseg = itisseg
        self.multi_modality = multi_modality
        self.seg_start_end = seg_start_end
        self.placehold = placehold

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
        if dataset_dict["height"] is None or dataset_dict["width"] is None:
            dataset_dict["height"], dataset_dict["width"], _ = image.shape

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
        gt_masks = dataset_dict["gt_masks"]

        class_text = f"{AUDIO_REF_START_TOKEN}{AUDIO_REF_TOKEN}{AUDIO_REF_END_TOKEN}"
        if self.placehold:
            class_text = class_text.replace(
                f"{AUDIO_REF_TOKEN}", f"{AUDIO_REF_TOKEN * 3}"
            )

        # Tag: multi-modality <seg_start>hilicopter with <audio_ref_start>audio_feat<audio_ref_end><seg_end>
        if self.multi_modality:
            class_text = f"{gt_classes[0]} with {class_text}"
        if self.seg_start_end:
            class_text = f"{SEG_START_TOKEN}{class_text}{SEG_END_TOKEN}"
        question = f"{DEFAULT_IMAGE_TOKEN}\nCan you segment object with the following audio {class_text} in this image?"
        if self.itisseg:
            answer = f"it is {self.obj_token}."
        else:
            answer = f"{gt_classes[0]}{self.obj_token}."

        conv = conversation_lib.conv_templates[self.conv_type].copy()

        conv.messages = []
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], answer)
        conversation = conv.get_prompt()

        audio = load_and_transform_audio_data(
            audio_paths=[dataset_dict["audio_file_name"]], device="cpu"
        )

        dataset_dict["clip_image"] = clip_image
        dataset_dict["sam_image"] = sam_image
        dataset_dict["sam_resized_size"] = sam_resized_size

        dataset_dict["question"] = question
        dataset_dict["conversation"] = conversation

        dataset_dict["gt_classes"] = gt_classes
        dataset_dict["gt_masks"] = gt_masks
        dataset_dict["gt_bbox"] = None

        dataset_dict["audio"] = audio

        return dataset_dict


class AVSMulti(Dataset):
    def __init__(
        self,
        split: str = "train",
        root_dir: str = "/mnt/data/data/avsbench/Multi-sources/",
        csv_file: str = "ms3_meta_data.csv",
        image_subdir: str = "ms3_data/visual_frames/",
        audio_subdir: str = "ms3_data/audio_wav/",
        mask_subdir: str = "ms3_data/gt_masks/",
        convert_classname: bool = True,
        original_resolution: bool = True,
    ):
        self.is_train = split == "train"
        self.root_dir = root_dir
        self.image_subdir = image_subdir
        self.audio_subdir = audio_subdir
        self.mask_subdir = mask_subdir
        self.convert_classname = convert_classname

        if original_resolution:
            self.image_subdir = image_subdir.replace(
                "visual_frames", "visual_frames_original_resolution"
            )

        df = pd.read_csv(os.path.join(root_dir, csv_file), sep=",")
        self.split = split
        self.df = df[df["split"] == self.split]

        print(f"avs_multi {split} has {self.__len__()} samples.")

    def __len__(self):
        if self.split == "train":
            return len(self.df) * 5
        elif self.split == "val":
            return len(self.df) * 5
        elif self.split == "test":
            return len(self.df) * 5
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        if self.split == "train":
            data = self.df.iloc[index // 5]
        elif self.split == "val":
            data = self.df.iloc[index // 5]
        elif self.split == "test":
            data = self.df.iloc[index // 5]

        video_name = data[0]
        # category = data[2]

        image_dir = os.path.join(self.root_dir, self.image_subdir, video_name)
        audio_dir = os.path.join(self.root_dir, self.audio_subdir, self.split)
        mask_dir = os.path.join(self.root_dir, self.mask_subdir, self.split, video_name)

        if self.split == "train":
            id = index % 5 + 1
        elif self.split == "val":
            id = index % 5 + 1
        elif self.split == "test":
            id = index % 5 + 1
        else:
            raise NotImplementedError

        # for id in ids:
        image_file = os.path.join(image_dir, f"{video_name}.mp4_{id}.png")
        assert os.path.exists(image_file), f"{image_file} does not exist"
        audio_file = os.path.join(audio_dir, f"{video_name}.wav")
        mask_file = os.path.join(mask_dir, f"{video_name}_{id}.png")

        gt_mask = Image.open(mask_file).convert(mode="P")
        gt_mask = (np.array(gt_mask) != 0).astype(np.uint8)
        # h, w = gt_mask.shape
        gt_mask = torch.as_tensor(gt_mask, dtype=torch.uint8).unsqueeze(0)

        # if self.convert_classname:
        #     category = avs_category_to_class[category]

        return dict(
            file_name=image_file,
            audio_file_name=audio_file,
            height=None,
            width=None,
            bbox=None,
            # gt_classes=torch.tensor(gt_classes),
            # gt_classes=[category],
            gt_classes=None,
            # gt_masks=torch.stack(gt_mask, dim=0),
            gt_masks=gt_mask,  # [1, h, w]
        )


class AVSMultiTokenized(AVSMulti):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        root_dir: str = "/mnt/data/data/avsbench/Multi-sources/",
        csv_file: str = "ms3_meta_data.csv",
        image_subdir: str = "ms3_data/visual_frames/",
        audio_subdir: str = "ms3_data/audio_wav/",
        mask_subdir: str = "ms3_data/gt_masks/",
        clip_version: str = "openai/clip-vit-large-patch14/",
        split: str = "train",
        image_size: int = 1024,  # for SAM
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        conv_type: str = "llava_v1",
        clip_resize_wo_crop: bool = True,
        convert_classname: bool = True,
        original_resolution: bool = True,
        itisseg: bool = False,
        multi_modality: bool = False,
        seg_start_end: bool = False,
        placehold: bool = False,
    ):
        super().__init__(
            split=split,
            root_dir=root_dir,
            csv_file=csv_file,
            image_subdir=image_subdir,
            audio_subdir=audio_subdir,
            mask_subdir=mask_subdir,
            convert_classname=convert_classname,
            original_resolution=original_resolution,
        )
        self.is_train = split == "train"
        self.tokenizer = tokenizer
        self.obj_token = "[SEG]"
        self.clip_processor = transformers.CLIPImageProcessor.from_pretrained(
            clip_version
        )

        self.conv_type = conv_type
        self.sam_image_size = image_size
        self.transforms = ResizeLongestSide(image_size)
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)

        self.clip_resize_wo_crop = clip_resize_wo_crop
        if self.clip_resize_wo_crop:
            self.clip_processor.do_center_crop = False

        self.itisseg = itisseg
        self.multi_modality = multi_modality
        self.seg_start_end = seg_start_end
        self.placehold = placehold

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
        if dataset_dict["height"] is None or dataset_dict["width"] is None:
            dataset_dict["height"], dataset_dict["width"], _ = image.shape

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
        gt_masks = dataset_dict["gt_masks"]

        class_text = f"{AUDIO_REF_START_TOKEN}{AUDIO_REF_TOKEN}{AUDIO_REF_END_TOKEN}"
        if self.placehold:
            class_text = class_text.replace(
                f"{AUDIO_REF_TOKEN}", f"{AUDIO_REF_TOKEN * 3}"
            )

        # Tag: multi-modality <seg_start>hilicopter with <audio_ref_start>audio_feat<audio_ref_end><seg_end>
        if self.multi_modality:
            class_text = f"{gt_classes[0]} with {class_text}"
        if self.seg_start_end:
            class_text = f"{SEG_START_TOKEN}{class_text}{SEG_END_TOKEN}"
        question = f"{DEFAULT_IMAGE_TOKEN}\nCan you segment object with the following audio {class_text} in this image?"
        # print(question)
        if self.itisseg:
            answer = f"it is {self.obj_token}."
        else:
            # answer = f"{gt_classes[0]}{self.obj_token}."
            answer = (
                f"it is {self.obj_token}."  # Tag: currently multi do not have gt class
            )

        conv = conversation_lib.conv_templates[self.conv_type].copy()

        conv.messages = []
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], answer)
        conversation = conv.get_prompt()

        audio = load_and_transform_audio_data(
            audio_paths=[dataset_dict["audio_file_name"]], device="cpu"
        )

        dataset_dict["clip_image"] = clip_image
        dataset_dict["sam_image"] = sam_image
        dataset_dict["sam_resized_size"] = sam_resized_size

        dataset_dict["question"] = question
        dataset_dict["conversation"] = conversation

        dataset_dict["gt_classes"] = gt_classes
        dataset_dict["gt_masks"] = gt_masks
        dataset_dict["gt_bbox"] = None

        dataset_dict["audio"] = audio

        return dataset_dict
