import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

import cv2
import torch
import torch.nn.functional as F
import transformers
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from model.llava import conversation as conversation_lib
from model.llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMG_REF_END_TOKEN,
    IMG_REF_NUM,
    IMG_REF_START_TOKEN,
    IMG_REF_TOKEN,
    SEG_END_TOKEN,
    SEG_START_TOKEN,
)
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide

from .coco_category import COCO_CATEGORIES

thing_dataset_id_to_name = {
    k["id"]: k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1
}
thing_ids = [k["id"] for k in COCO_CATEGORIES if k["isthing"] == 1]


@dataclass
class DataCollector(object):
    def __init__(
        self,
        tokenizer=None,
        conv_type="llava_v1",
        use_mm_start_end=True,
        left_pad=False,
        roi=False,
    ):
        self.tokenizer = tokenizer
        self.conv_type = conv_type
        self.use_mm_start_end = use_mm_start_end
        self.left_pad = left_pad
        self.roi = roi

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        clip_images, sam_images, sam_resized_sizes = [], [], []
        gt_classes, gt_masks, gt_bboxes = [], [], []
        conversations, questions = [], []
        file_names, heights, widths, image_ids = [], [], [], []
        audios = []
        ref_images = []
        for instance in instances:
            clip_images.append(instance["clip_image"])
            sam_images.append(instance["sam_image"])
            sam_resized_sizes.append(instance["sam_resized_size"])

            gt_classes.append(instance["gt_classes"])
            gt_masks.append(instance["gt_masks"])
            gt_bboxes.append(instance["gt_bbox"])

            # print(instance["conversation"])
            conversations.append(instance["conversation"])
            questions.append(instance["question"])

            file_names.append(instance["file_name"])
            heights.append(instance["height"])
            widths.append(instance["width"])

            # if "image_id" in instance:
            #     image_ids.append(instance["image_id"])
            # else:
            #     image_ids.append(None)

            if "audio" in instance:
                audios.append(instance["audio"])
            else:
                audios.append(None)
                # pass

            if "ref_image" in instance:
                if self.roi:
                    ref_images.append(instance["gt_bbox"])
                else:
                    ref_images.append(instance["ref_image"])
            else:
                ref_images.append(None)
                # pass

        if self.use_mm_start_end:
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            for i in range(len(conversations)):
                conversations[i] = conversations[i].replace(
                    DEFAULT_IMAGE_TOKEN, replace_token
                )

        input_ids = [
            # tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt") for prompt in conversations
            tokenizer_image_token(
                prompt,
                self.tokenizer,
                return_tensors="pt",
                placehold=True,
            )
            for prompt in conversations
        ]

        # print([i for i in input_ids])
        if self.left_pad:
            # only take ids before the second ":"(29901)
            for i in range(len(input_ids)):
                colon_id = torch.where(input_ids[i] == 29901)[0][1]
                input_ids[i] = input_ids[i][: colon_id + 1]
            # print([i for i in input_ids])

            lens = [t.shape[0] for t in input_ids]
            max_len = max(lens)
            for i in range(len(input_ids)):
                if lens[i] < max_len:
                    input_ids[i] = torch.cat(
                        [
                            torch.ones(max_len - lens[i], dtype=torch.long)
                            * self.tokenizer.pad_token_id,
                            input_ids[i],
                        ]
                    )
                    # input_ids[ii] = [self.tokenizer.pad_token_id] * (max_len - len(input_ids[ii])) + input_ids[ii]
            # input_ids = torch.tensor(input_ids, dtype=torch.long)
            input_ids = torch.stack(input_ids, dim=0)
            attention_masks = input_ids.ne(self.tokenizer.pad_token_id)

            return dict(
                clip_images=torch.stack(clip_images, dim=0),
                sam_images=torch.stack(sam_images, dim=0),
                sam_resized_sizes=sam_resized_sizes,
                input_ids=input_ids,
                # labels=targets,
                attention_masks=attention_masks,
                gt_classes=gt_classes,
                gt_masks=gt_masks,
                gt_bboxes=gt_bboxes,
                file_name=file_names,
                height=heights,
                width=widths,
                image_id=image_ids,
                audios=audios,
                ref_images=ref_images,
            )

        else:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
        # print(input_ids)
        attention_masks = input_ids.ne(self.tokenizer.pad_token_id)

        # conv = conversation_lib.default_conversation.copy()
        conv = conversation_lib.conv_templates[self.conv_type].copy()
        targets = input_ids.clone()

        if self.conv_type == "llava_v1":
            sep = conv.sep + conv.roles[1] + ": "  # "###Assistant: "
        else:
            raise NotImplementedError

        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(self.tokenizer.pad_token_id).sum())
            # print("total_len: ", total_len)

            rounds = conversation.split(conv.sep2)
            # print("conv.sep2: ", conv.sep2)
            # print("rounds: ", rounds)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                # print("parts: ", parts)
                assert len(parts) == 2, (len(parts), rou)
                parts[0] += sep

                if DEFAULT_IMAGE_TOKEN in conversation:
                    # print("rou: ", rou)
                    # print("parts[0]: ", parts[0])
                    round_len = len(
                        tokenizer_image_token(rou, self.tokenizer, placehold=True)
                    )
                    instruction_len = (
                        len(
                            tokenizer_image_token(
                                parts[0], self.tokenizer, placehold=True
                            )
                        )
                        - 2
                    )
                    # print("len: ", round_len, instruction_len)
                else:
                    round_len = len(self.tokenizer(rou).input_ids)
                    instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX

            # Tag: why???
            if cur_len < self.tokenizer.model_max_length:
                assert cur_len == total_len

        return dict(
            clip_images=torch.stack(clip_images, dim=0),
            sam_images=torch.stack(sam_images, dim=0),
            sam_resized_sizes=sam_resized_sizes,
            input_ids=input_ids,
            labels=targets,
            attention_masks=attention_masks,
            gt_classes=gt_classes,
            gt_masks=gt_masks,
            gt_bboxes=gt_bboxes,
            file_name=file_names,
            height=heights,
            width=widths,
            # image_id=image_ids if len(image_ids) > 0 else None,
            # audios=torch.cat(audios, dim=0) if len(audios) > 0 else None,
            # ref_images=torch.stack(ref_images, dim=0) if len(ref_images) > 0 else None,
            image_id=image_ids,
            audios=audios,
            ref_images=ref_images,
        )


class COCOIns(Dataset):
    def __init__(
        self,
        is_train: bool = True,
        image_root: str = "/mnt/data/data/coco/train2017/",
        json_file: str = "/mnt/data/data/coco/annotations/instances_train2017.json",
        filter_area: float = None,
        overfit: bool = False,
    ):
        if not is_train:
            image_root = "/mnt/data/data/coco/val2017/"
            json_file = "/mnt/data/data/coco/annotations/instances_val2017.json"

        self.coco = COCO(json_file)
        self.root_dir = image_root
        # self.transforms = transforms
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.filter_area = filter_area

        # samples without gt_masks
        ignore_index = []
        for i in range(len(self.ids)):
            anns = self.coco.loadAnns(self.coco.getAnnIds(self.ids[i]))
            if len(anns) == 0:
                ignore_index.append(i)
        for i in reversed(ignore_index):
            self.ids.remove(self.ids[i])

        # Overfitting test
        if overfit:
            self.ids = [self.ids[:1]] * 40000

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        image_info = self.coco.loadImgs(image_id)[0]

        file_name = os.path.join(self.root_dir, image_info["file_name"])
        height = image_info["height"]
        width = image_info["width"]

        anns = self.coco.loadAnns(self.coco.getAnnIds(image_id))
        bbox, gt_classes, gt_masks = [], [], []
        for ann in anns:
            if ann["iscrowd"] != 0:
                continue

            polygon = ann["segmentation"]
            rles = coco_mask.frPyObjects(polygon, height, width)
            mask = coco_mask.decode(rles)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = torch.as_tensor(mask, dtype=torch.uint8)
            mask = mask.any(dim=2)
            if self.filter_area is not None:
                if (
                    mask.sum().item() / (mask.shape[0] * mask.shape[1])
                    < self.filter_area
                ):
                    continue
            gt_masks.append(mask)

            bbox.append(ann["bbox"])  # [x, y, w, h] [x,y]=top-left
            gt_classes.append(ann["category_id"])
        if len(gt_masks) == 0:
            return None

        return dict(
            image_id=image_id,
            file_name=file_name,
            height=height,
            width=width,
            bbox=torch.tensor(
                bbox, dtype=torch.float32
            ),  # [N, 4], [x, y, w, h] [x,y]=top-left
            gt_classes=torch.tensor(gt_classes),
            gt_masks=torch.stack(gt_masks, dim=0),
        )


class COCOInstanceTokenized(COCOIns):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        clip_version: str = "openai/clip-vit-large-patch14/",
        is_train: bool = True,
        image_root: str = "/mnt/data/data/coco/train2017/",
        json_file: str = "/mnt/data/data/coco/annotations/instances_train2017.json",
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
    ):
        super().__init__(
            is_train, image_root, json_file, filter_area=filter_area, overfit=overfit
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
            # dataset_dict = super().__getitem__(0)

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

        orig_gt_bbox = dataset_dict["bbox"]  # [N, 4], [x, y, w, h] [x,y]=top-left
        # convert xywh to xyxy
        orig_gt_bbox[:, 2:] += orig_gt_bbox[
            :, :2
        ]  # [x1, y1, x2, y2]  top-left, bottom-right
        gt_bbox = orig_gt_bbox.clone()
        # normalize to [0, 1]
        gt_bbox[:, 0::2] /= dataset_dict["width"]
        gt_bbox[:, 1::2] /= dataset_dict["height"]
        # keep 2 decimals
        gt_bbox = torch.round(gt_bbox * 100).long()

        # if not self.is_train:
        #     dataset_dict["blip2_image"] = blip2_image
        #     dataset_dict["sam_image"] = sam_image
        #     dataset_dict["sam_resized_size"] = sam_resized_size
        #     gt_classes = dataset_dict["gt_classes"]
        #     gt_masks = dataset_dict["gt_masks"]
        #     dataset_dict = self.retrieve_eval_data(dataset_dict)
        #     return dataset_dict

        unique_classes = set(gt_classes.tolist())
        # print("unique_classes: ", unique_classes)   # {49, 50, 45}

        unexist = False
        if random.random() < 0.0:
            # Tag: randomly sample a class that does not exist in the image
            unexist = True
            sampled_class_num = 1
            unexist_classes = [
                k for k in thing_dataset_id_to_name.keys() if k not in unique_classes
            ]
            sampled_classes = random.sample(unexist_classes, sampled_class_num)
        else:
            # Tag: normal sampling
            # sampled_class_num = random.randint(1, self.sampled_class_num)
            sampled_class_num = self.sampled_class_num
            if sampled_class_num > len(unique_classes):
                sampled_class_num = len(unique_classes)
            # sampled_class_num = 1   # Tag: only select 1 class per image
            sampled_classes = random.sample(unique_classes, sampled_class_num)

        if self.all:
            sampled_class_num = len(unique_classes)
            sampled_classes = list(unique_classes)

        # Tag: overfitting test
        # sampled_class_num = 3
        # sampled_classes = unique_classes
        # print("sampled_classes: ", sampled_classes) # [49]

        gt_classes_text, gt_classes_text_obj = [], []
        sampled_class, sampled_mask, sampled_bbox = [], [], []
        for c in sampled_classes:
            sampled_idx = gt_classes == c

            # Tag: if using multiple object tokens
            if self.num_obj_token > 1 and sampled_idx.sum() > self.num_obj_token:
                sum_i = 0
                for i in range(sampled_idx.shape[0]):
                    if sum_i == self.num_obj_token:
                        sampled_idx[i:] = False
                        break
                    if sampled_idx[i]:
                        sum_i += 1

            sampled_class.append(gt_classes[sampled_idx])
            sampled_mask.append(gt_masks[sampled_idx])
            sampled_bbox.append(orig_gt_bbox[sampled_idx])
            sampled_bbox_ = gt_bbox[sampled_idx]

            # class_text = thing_dataset_id_to_name[contiguous_id_to_thing_dataset_id[c]]
            class_text = thing_dataset_id_to_name[c]
            gt_classes_text.append(class_text)

            instance_num = int(sampled_idx.sum())
            # Tag: if using multiple object tokens
            if self.num_obj_token > 1:
                # obj_tokens = self.obj_token[(random.sample(range(self.num_obj_token), instance_num))]
                obj_tokens = random.sample(self.obj_token, instance_num)
                gt_classes_text_obj.append(f"{class_text}{''.join(obj_tokens)}")
            else:
                if self.with_bbox:  # person [x1][y1][x2][y2][seg], person...
                    text = []
                    for i in range(instance_num):
                        bbox = sampled_bbox_[i]
                        loc_tokens = [f"[LOC{int(j)}]" for j in bbox]
                        # print(loc_tokens)
                        text.append(
                            f"{class_text}{''.join(loc_tokens)}{self.obj_token}"
                        )
                    gt_classes_text_obj.append(",".join(text))
                else:
                    text = []
                    for i in range(instance_num):
                        text.append(
                            f"{class_text}{self.obj_token}"
                        )  # person[seg],person[seg],cat[seg]
                    gt_classes_text_obj.append(",".join(text))
                    # gt_classes_text_obj.append(f"{class_text}{self.obj_token * instance_num}")
                    # print(gt_classes_text_obj)

        # gt_classes = torch.cat(sampled_class, dim=0)
        gt_classes = gt_classes_text
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

    def retrieve_eval_data(self, dataset_dict):
        gt_classes = dataset_dict["gt_classes"]
        # gt_masks = dataset_dict["gt_masks"]

        unique_classes = set(gt_classes.tolist())
        unique_classnames = [thing_dataset_id_to_name[c] for c in unique_classes]

        prompts = []
        for c in unique_classnames:
            prompt = f"Question: Can you segment {c} in this image? Answer:"
            prompts.append(prompt)
        dataset_dict["prompts"] = prompts
        return dataset_dict


class COCOSemantic(COCOIns):
    def __init__(
        self,
        is_train: bool = True,
        image_root: str = "/mnt/data/data/coco/train2017/",
        json_file: str = "/mnt/data/data/coco/annotations/instances_train2017.json",
        filter_area: float = None,
        overfit: bool = False,
        sample_by_category: bool = False,
        box_img_threshold: float = 0.1,
        mask_box_threshold: float = 0.4,
        num_sample: int = 1,
        ref_json_file: str = "samples_0.05_0.25.json",
        apply_mask: float = 0,
    ):
        super().__init__(
            is_train, image_root, json_file, filter_area=filter_area, overfit=overfit
        )
        self.sample_by_category = sample_by_category
        if sample_by_category:
            self.samples_by_category = {}
            self.box_img_threshold = box_img_threshold
            self.mask_box_threshold = mask_box_threshold

        self.num_sample = num_sample
        with open(ref_json_file, "r") as f:
            self.samples = json.load(f)
        # too many "person", sample 10% of them
        self.samples[str(1)] = random.sample(
            self.samples[str(1)], int(len(self.samples[str(1)]) * 0.1)
        )

        self.apply_mask = apply_mask

    def __getitem__(self, index, add_sample=False):
        data_dict = super().__getitem__(index)

        gt_classes = data_dict["gt_classes"]
        gt_masks = data_dict["gt_masks"]
        gt_bbox = data_dict["bbox"]
        sem_gt_classes, sem_gt_masks, sem_gt_bbox = [], [], []
        sem_has_bbox = []
        for c in gt_classes.unique():
            idx = gt_classes == c
            if idx.sum() == 1:
                sem_gt_bbox.append(gt_bbox[idx][0])  # [4]
                sem_has_bbox.append(True)

                if self.sample_by_category and add_sample:
                    box = gt_bbox[idx]
                    mask = gt_masks[idx]
                    box_area = box[0][-1] * box[0][-2]
                    img_area = data_dict["height"] * data_dict["width"]
                    if (
                        box_area / img_area > self.box_img_threshold
                        and mask.sum() / box_area > self.mask_box_threshold
                    ):
                        # c = c.item()
                        if c.item() not in self.samples_by_category:
                            self.samples_by_category[c.item()] = []
                        self.samples_by_category[c.item()].append(
                            (index, int(torch.where(idx)[0][0]))
                        )
                        # self.samples_by_category[c.item()].append((index, int()))
            else:
                sem_gt_bbox.append(torch.zeros_like(gt_bbox[0]))
                sem_has_bbox.append(False)
            sem_mask = gt_masks[idx]  # [N, H, W]
            sem_mask = torch.any(sem_mask, dim=0)  # [H, W]
            sem_gt_classes.append(c)
            sem_gt_masks.append(sem_mask)

        sem_gt_classes = torch.stack(sem_gt_classes, dim=0)
        sem_gt_masks = torch.stack(sem_gt_masks, dim=0)
        sem_gt_bbox = torch.stack(sem_gt_bbox, dim=0)
        sem_has_bbox = torch.tensor(sem_has_bbox)

        if self.num_sample == 1:
            sample_idx = random.randint(0, sem_gt_classes.shape[0] - 1)
            sem_gt_classes = sem_gt_classes[sample_idx]
            sem_gt_masks = sem_gt_masks[sample_idx]
            sem_gt_bbox = sem_gt_bbox[sample_idx]

            refs = self.samples[str(sem_gt_classes.item())]
            ref_idx = random.choice(refs)
            # print(ref_idx)
            ref_data_dict = super().__getitem__(ref_idx[0])
            ref_img = cv2.imread(ref_data_dict["file_name"])
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

            if self.apply_mask > 0:
                if random.random() > self.apply_mask:
                    ref_gt_mask = ref_data_dict["gt_masks"][ref_idx[1]]
                    ref_img = ref_img * ref_gt_mask.unsqueeze(-1).numpy()

            x1, y1, w, h = ref_data_dict["bbox"][ref_idx[1]].int().tolist()
            ref_img = ref_img[y1 + 1 : y1 + h - 1, x1 + 1 : x1 + w - 1]

            data_dict.update(
                {
                    "gt_classes": sem_gt_classes,
                    "gt_masks": sem_gt_masks,
                    "ref_img": ref_img,
                    "gt_bbox": sem_gt_bbox,
                }
            )
            return data_dict

        # data_dict.update({
        #     "sem_gt_classes": sem_gt_classes,
        #     "sem_gt_masks": sem_gt_masks,
        #     "sem_gt_bbox": sem_gt_bbox,
        #     "sem_has_bbox": sem_has_bbox,
        # })
        data_dict.update(
            {
                "gt_classes": sem_gt_classes,
                "gt_masks": sem_gt_masks,
                "bbox": sem_gt_bbox,
                "sem_has_bbox": sem_has_bbox,
            }
        )
        return data_dict


class COCOSemantic_inv_Tokenized(COCOSemantic):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        clip_version: str = "openai/clip-vit-large-patch14/",
        is_train: bool = True,
        image_root: str = "/mnt/data/data/coco/train2017/",
        json_file: str = "/mnt/data/data/coco/annotations/instances_train2017.json",
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
        placehold: bool = False,  # put 4 * <ref_img>
        apply_mask: float = 0,
    ):
        super().__init__(
            is_train,
            image_root,
            json_file,
            filter_area=filter_area,
            overfit=overfit,
            apply_mask=apply_mask,
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
        self.placehold = placehold

        # self.apply_mask = apply_mask

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

        ref_image = dataset_dict["ref_img"]
        if self.clip_resize_wo_crop:
            ref_image = cv2.resize(ref_image, (224, 224))
        ref_image = self.clip_processor.preprocess(ref_image, return_tensors="pt")[
            "pixel_values"
        ][0]

        gt_class = thing_dataset_id_to_name[gt_classes.item()]
        class_obj_text = f"{gt_class}{self.obj_token}"

        class_text = f"{IMG_REF_START_TOKEN}{IMG_REF_TOKEN}{IMG_REF_END_TOKEN}"
        if self.placehold:
            class_text = class_text.replace(
                f"{IMG_REF_TOKEN}", f"{IMG_REF_TOKEN * IMG_REF_NUM}"
            )

        if self.seg_start_end:
            class_text = f"{SEG_START_TOKEN}{class_text}{SEG_END_TOKEN}"

        question = f"{DEFAULT_IMAGE_TOKEN}\nCan you segment object with the following reference {class_text} in this image?"
        answer = f"{class_obj_text}."

        conv = conversation_lib.conv_templates[self.conv_type].copy()

        conv.messages = []
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], answer)
        conversation = conv.get_prompt()

        dataset_dict["clip_image"] = clip_image
        dataset_dict["sam_image"] = sam_image
        dataset_dict["sam_resized_size"] = sam_resized_size

        dataset_dict["question"] = question
        dataset_dict["conversation"] = conversation

        dataset_dict["gt_classes"] = [gt_class]
        dataset_dict["gt_masks"] = gt_masks.unsqueeze(0)
        # dataset_dict["gt_bbox"] = gt_bbox

        dataset_dict["ref_image"] = ref_image
        # print(conversation)

        return dataset_dict
