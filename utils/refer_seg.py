import os
import random
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from pycocotools import mask
from torch.utils.data import Dataset

from model.llava import conversation as conversation_lib
from model.llava.constants import DEFAULT_IMAGE_TOKEN, SEG_END_TOKEN, SEG_START_TOKEN
from model.segment_anything.utils.transforms import ResizeLongestSide

from .grefer import G_REFER
from .refer import REFER

DATASET_LABEL = {
    "refcoco": 0,
    "refcoco+": 1,
    "refcocog": 2,
    "refclef": 3,
    "grefcoco": 4,
}


class REFCOCO(Dataset):
    def __init__(
        self,
        data_root: str = "/mnt/data/data/",
        is_train: bool = True,
        datasets: str = "refcoco,refcoco+,refcocog,refclef",  # grefcoco
        use2017: bool = True,
        all: bool = False,
        split: str = None,
    ):
        self.is_train = is_train
        self.all = all
        self.datasets = datasets

        self.ref_dataset_label = []  # 0: refcoco, 1: refcoco+, 2: refcocog, 3: refclef
        self.images = []
        self.annotations = {}
        self.img2refs = {}

        datasets = datasets.split(",")
        for dataset in datasets:
            splitBy = "umd" if dataset == "refcocog" else "unc"

            if dataset == "grefcoco":
                refer_api = G_REFER(
                    data_root=data_root,
                    dataset=dataset,
                    splitBy=splitBy,
                    use2017=use2017,
                )
            else:
                refer_api = REFER(
                    data_root=data_root,
                    dataset=dataset,
                    splitBy=splitBy,
                    use2017=use2017,
                )

            if split is None:
                if is_train:
                    split = "train"
                else:
                    split = "val"
            else:
                pass
            # split = "train" if is_train else "val"
            ref_ids = refer_api.getRefIds(split=split)
            print(len(ref_ids))
            images_ids = refer_api.getImgIds(ref_ids=ref_ids)
            refs = refer_api.loadRefs(ref_ids=ref_ids)

            loaded_images = refer_api.loadImgs(image_ids=images_ids)

            for item in loaded_images:
                item = item.copy()
                if dataset == "refclef":
                    root_prefix = os.path.join(data_root, "saiapr_tc-12")
                    item["file_name"] = os.path.join(root_prefix, item["file_name"])
                else:
                    if use2017:
                        root_prefix = os.path.join(data_root, "coco/train2017")
                    else:
                        raise NotImplementedError
                    item["file_name"] = os.path.join(
                        root_prefix, f"{str(item['id']).zfill(12)}.jpg"
                    )
                self.images.append(item)
                self.ref_dataset_label.append(DATASET_LABEL[dataset])

            self.annotations[DATASET_LABEL[dataset]] = refer_api.Anns

            img2refs = {}
            for ref in refs:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [ref]
            self.img2refs[DATASET_LABEL[dataset]] = img2refs

        print(f"ref dataset has {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_info = self.images[index]
        image_id = image_info["id"]
        image_path = image_info["file_name"]

        dataset_label = self.ref_dataset_label[index]
        refs = self.img2refs[dataset_label][image_id]
        # print(refs)

        sents, ann_ids = [], []
        if self.is_train:
            for ref in refs:
                if self.all:
                    for sent in ref["sentences"]:
                        text = sent["sent"]
                        sents.append(text.strip().lower())
                        ann_ids.append(ref["ann_id"])
                else:
                    sent_rand = random.choice(
                        ref["sentences"]
                    )  # randomly choose one sentence for each mask
                    text = sent_rand["sent"]
                    sents.append(text.strip().lower())
                    ann_ids.append(ref["ann_id"])
        else:
            for ref in refs:
                for sent in ref["sentences"]:
                    sents.append(sent["sent"].strip().lower())
                    ann_ids.append(ref["ann_id"])
        # print(sents)

        masks = []
        boxes = []
        for ann_id in ann_ids:
            if isinstance(ann_id, list):
                # flag = True
                if -1 in ann_id:
                    assert len(ann_id) == 1
                    m = np.zeros((image_info["height"], image_info["width"])).astype(
                        np.uint8
                    )
                    # box = np.zeros((4,)).astype(np.uint8)
                else:
                    m_final = np.zeros(
                        (image_info["height"], image_info["width"])
                    ).astype(np.uint8)
                    for ann_id_i in ann_id:
                        ann = self.annotations[dataset_label][ann_id_i]

                        if len(ann["segmentation"]) == 0:
                            m = np.zeros(
                                (image_info["height"], image_info["width"])
                            ).astype(np.uint8)
                        else:
                            if isinstance(ann["segmentation"][0], list):  # polygon
                                rle = mask.frPyObjects(
                                    ann["segmentation"],
                                    image_info["height"],
                                    image_info["width"],
                                )
                            else:
                                rle = ann["segmentation"]
                                for i in range(len(rle)):
                                    if not isinstance(rle[i]["counts"], bytes):
                                        rle[i]["counts"] = rle[i]["counts"].encode()
                            m = mask.decode(rle)
                            m = np.sum(
                                m, axis=2
                            )  # sometimes there are multiple binary map (corresponding to multiple segs)
                            m = m.astype(np.uint8)  # convert to np.uint8
                        m_final = m_final | m
                    m = m_final
                    # box =
                masks.append(m)
                # boxes.append(box)
                continue

            ann = self.annotations[dataset_label][ann_id]
            # print(ann_id, ann)
            box = ann["bbox"]  # [x, y, w, h]
            # print(box)
            box[2] += box[0]
            box[3] += box[1]
            boxes.append(np.array(box))

            if len(ann["segmentation"]) == 0:
                m = np.zeros((image_info["height"], image_info["width"])).astype(
                    np.uint8
                )
                masks.append(m)
                continue

            if isinstance(ann["segmentation"][0], list):  # polygon
                rle = mask.frPyObjects(
                    ann["segmentation"], image_info["height"], image_info["width"]
                )
            else:
                rle = ann["segmentation"]
                for i in range(len(rle)):
                    if not isinstance(rle[i]["counts"], bytes):
                        rle[i]["counts"] = rle[i]["counts"].encode()
            m = mask.decode(rle)
            m = np.sum(
                m, axis=2
            )  # sometimes there are multiple binary map (corresponding to multiple segs)
            m = m.astype(np.uint8)  # convert to np.uint8
            masks.append(m)

        masks = np.stack(masks, axis=0)
        try:
            boxes = np.stack(boxes, axis=0)
        except:
            boxes = None

        if self.datasets == "grefcoco":
            mask_sum = masks.sum(axis=(-1, -2))
            empty_mask_idx = np.where(mask_sum == 0)[0][0]
            sents.pop(empty_mask_idx)
            masks = np.delete(masks, empty_mask_idx, axis=0)
            print(empty_mask_idx)

        return dict(
            file_name=image_path,
            height=image_info["height"],
            width=image_info["width"],
            gt_classes=sents,
            gt_masks=torch.from_numpy(masks),
            gt_bbox=torch.from_numpy(boxes).int() if boxes is not None else None,
        )


class REFCOCOTokenized(REFCOCO):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_root: str = "/mnt/data/data/",
        datasets: str = "refcoco,refcoco+,refcocog",  # ,refclef",
        use2017: bool = True,
        clip_version: str = "openai/clip-vit-large-patch14/",
        is_train: bool = True,
        image_size: int = 1024,  # for SAM
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        filter_area: float = None,
        all: bool = False,
        conv_type: str = "llava_v1",
        num_obj_token: int = 1,
        with_bbox: bool = False,
        clip_resize_wo_crop: bool = True,
        sampled_class_num: int = 1,
        overfit: bool = False,
        itisseg: bool = False,
        seg_start_end: bool = False,
        img_ref: bool = False,
        split: str = None,
    ):
        super().__init__(
            data_root=data_root,
            is_train=is_train,
            datasets=datasets,
            use2017=use2017,
            split=split,
        )
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
        self.img_ref = img_ref

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

        unexist = False
        if random.random() < 0.0:
            # Tag: randomly sample a class that does not exist in the image
            unexist = True
            sampled_class_num = 1
            # unexist_classes = [k for k in thing_dataset_id_to_name.keys() if k not in unique_classes]
            # sampled_classes = random.sample(unexist_classes, sampled_class_num)
        else:
            if self.is_train:
                # Tag: normal sampling
                sampled_class_num = random.randint(1, self.sampled_class_num)
                # sampled_class_num = self.sampled_class_num
                if sampled_class_num > len(gt_classes):
                    sampled_class_num = len(gt_classes)
                # sampled_class_num = 1   # Tag: only select 1 class per image

                sampled_idxs = random.sample(range(len(gt_classes)), sampled_class_num)
                # sampled_classes = gt_classes[sampled_idxs]
            else:
                raise NotImplementedError

        if self.all:
            sampled_class_num = len(gt_classes)
            sampled_classes = gt_classes

        gt_classes_text, gt_classes_text_obj = [], []
        sampled_class, sampled_mask, sampled_bbox = [], [], []
        for sample_idx in sampled_idxs:
            sampled_class.append(gt_classes[sample_idx])
            sampled_mask.append(gt_masks[sample_idx])
            # sampled_bbox.append(dataset_dict["gt_bbox"][sample_idx])

            class_text = gt_classes[sample_idx]
            gt_classes_text_obj.append(f"{class_text}{self.obj_token}")

        gt_classes = sampled_class
        gt_masks = torch.stack(sampled_mask, dim=0)

        if sampled_class_num == 1:
            class_text = sampled_class[0]
            class_obj_text = gt_classes_text_obj[0]
        else:
            # class_text = ", ".join(sampled_class[:-1]) + f" and {sampled_class[-1]}"
            # class_obj_text = ", ".join(gt_classes_text_obj[:-1]) + f" and {gt_classes_text_obj[-1]}"
            class_text = ", ".join(sampled_class)
            class_obj_text = ", ".join(gt_classes_text_obj)

        if self.seg_start_end:
            class_text = f"{SEG_START_TOKEN}{class_text}{SEG_END_TOKEN}"

        question = f"{DEFAULT_IMAGE_TOKEN}\nCan you segment {class_text} in this image?"
        if not unexist:
            answer = f"{class_obj_text}."
            if self.itisseg:
                answer = f"it is {self.obj_token}."
        else:
            answer = f"there is no {class_text} in this image."

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


class REFCOCO_val(Dataset):
    def __init__(
        self,
        data_root: str = "/mnt/data/data/",
        split: str = "val",  # "testA", "testB"
        dataset: str = "refcoco",  # refcoco+,refcocog,refclef", #grefcoco
        use2017: bool = True,
        all: bool = False,
    ):
        self.data_root = data_root
        splitBy = "umd" if dataset == "refcocog" else "unc"
        if dataset == "grefcoco":
            self.refer_api = G_REFER(
                data_root=data_root, dataset=dataset, splitBy=splitBy, use2017=use2017
            )
        else:
            self.refer_api = REFER(
                data_root=data_root, dataset=dataset, splitBy=splitBy, use2017=use2017
            )

        self.ref_ids = self.refer_api.getRefIds(split=split)
        self.annotations = self.refer_api.Anns

        print(f"{dataset}_{split} dataset has {self.__len__()} images")

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        ref_id = self.ref_ids[index]
        img_id = self.refer_api.getImgIds(ref_id)
        image_info = self.refer_api.Imgs[img_id[0]]
        image_path = image_info["file_name"][-16:]
        image_path = os.path.join(self.data_root, "coco/train2017", image_path)
        assert os.path.exists(image_path)

        h, w = image_info["height"], image_info["width"]

        ref = self.refer_api.loadRefs(ref_id)
        ann_id = ref[0]["ann_id"]

        ann = self.annotations[ann_id]
        m = np.zeros((h, w)).astype(np.uint8)
        if isinstance(ann["segmentation"][0], list):  # polygon
            rle = mask.frPyObjects(ann["segmentation"], h, w)
        else:
            rle = ann["segmentation"]
            for i in range(len(rle)):
                if not isinstance(rle[i]["counts"], bytes):
                    rle[i]["counts"] = rle[i]["counts"].encode()
        m = mask.decode(rle)
        m = np.sum(
            m, axis=2
        )  # sometimes there are multiple binary map (corresponding to multiple segs)
        m = m.astype(np.uint8)  # convert to np.uint8

        # Tag: currently only use the last reference sentence
        sent = ref[0]["sentences"][-1]["sent"]

        return dict(
            file_name=image_path,
            height=image_info["height"],
            width=image_info["width"],
            gt_classes=[sent],
            gt_masks=torch.from_numpy(m).unsqueeze(0),
        )


class REFCOCO_valTokenized(REFCOCO_val):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_root: str = "/mnt/data/data/",
        dataset: str = "refcoco",  # ,refcoco+,refcocog", #,refclef",
        use2017: bool = True,
        clip_version: str = "openai/clip-vit-large-patch14/",
        split: str = "val",  # "testA", "testB"
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
    ):
        super().__init__(
            data_root=data_root, split=split, dataset=dataset, use2017=use2017
        )
        # self.is_train = is_train

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
        gt_masks = dataset_dict["gt_masks"]

        class_text = gt_classes[0]
        # print(f"{index}: {class_text}")
        class_obj_text = f"{class_text}{self.obj_token}"

        unexist = False
        # if random.random() < 0.0:
        #     # Tag: randomly sample a class that does not exist in the image
        #     unexist = True
        #     sampled_class_num = 1
        #     # unexist_classes = [k for k in thing_dataset_id_to_name.keys() if k not in unique_classes]
        #     # sampled_classes = random.sample(unexist_classes, sampled_class_num)
        # else:
        #     if self.is_train:
        #         # Tag: normal sampling
        #         sampled_class_num = random.randint(1, self.sampled_class_num)
        #         # sampled_class_num = self.sampled_class_num
        #         if sampled_class_num > len(gt_classes):
        #             sampled_class_num = len(gt_classes)
        #         # sampled_class_num = 1   # Tag: only select 1 class per image

        #         sampled_idxs = random.sample(range(len(gt_classes)), sampled_class_num)
        #         # sampled_classes = gt_classes[sampled_idxs]
        #     else:
        #         raise NotImplementedError

        # if self.all:
        #     sampled_class_num = len(gt_classes)
        #     sampled_classes = gt_classes

        # gt_classes_text, gt_classes_text_obj = [], []
        # sampled_class, sampled_mask, sampled_bbox = [], [], []
        # for sample_idx in sampled_idxs:
        #     sampled_class.append(gt_classes[sample_idx])
        #     sampled_mask.append(gt_masks[sample_idx])

        #     class_text = gt_classes[sample_idx]
        #     gt_classes_text_obj.append(f"{class_text}{self.obj_token}")

        # gt_classes = sampled_class
        # gt_masks = torch.stack(sampled_mask, dim=0)

        # if sampled_class_num == 1:
        #     class_text = sampled_class[0]
        #     class_obj_text = gt_classes_text_obj[0]
        # else:
        #     # class_text = ", ".join(sampled_class[:-1]) + f" and {sampled_class[-1]}"
        #     # class_obj_text = ", ".join(gt_classes_text_obj[:-1]) + f" and {gt_classes_text_obj[-1]}"
        #     class_text = ", ".join(sampled_class)
        #     class_obj_text = ", ".join(gt_classes_text_obj)

        if self.seg_start_end:
            class_text = f"{SEG_START_TOKEN}{class_text}{SEG_END_TOKEN}"
        question = f"{DEFAULT_IMAGE_TOKEN}\nCan you segment {class_text} in this image?"
        if not unexist:
            # target = f"{class_obj_text}.</s>"
            answer = f"{class_obj_text}."
        else:
            answer = f"there is no {class_text} in this image."

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
