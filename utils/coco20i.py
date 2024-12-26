import os
import os.path
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

import cv2
import torch
import torch.nn.functional as F
import transformers

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
from utils.coco_instance import COCOSemantic, thing_dataset_id_to_name

COCO_BASE = "/mnt/data/data/coco/"
COCO20i_BASE = "/mnt/data/data/coco_20i/"

id2id = {}  # [0-80](coco 80) - [1-90](orig id)
id2id_rev = {}
for i, id in enumerate(list(thing_dataset_id_to_name.keys())):
    id2id[i + 1] = id
    id2id_rev[id] = i + 1
coco20_split = {}
for i in range(4):
    if i == 0:
        ids = list(range(1, 78, 4))
    elif i == 1:
        ids = list(range(2, 79, 4))
    elif i == 2:
        ids = list(range(3, 80, 4))
    elif i == 3:
        ids = list(range(4, 81, 4))

    l = []  # noqa: E741
    for ii in ids:
        l.append(id2id[ii])
    coco20_split[i] = l


@dataclass
class DataCollector(object):
    def __init__(
        self,
        tokenizer=None,
        conv_type="llava_v1",
        use_mm_start_end=True,
        left_pad=False,
    ):
        self.tokenizer = tokenizer
        self.conv_type = conv_type
        self.use_mm_start_end = use_mm_start_end
        self.left_pad = left_pad

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        clip_images, sam_images, sam_resized_sizes = [], [], []
        gt_classes, gt_masks, gt_bboxes = [], [], []
        conversations, questions = [], []
        file_names, heights, widths, image_ids = [], [], [], []
        audios = []
        ref_images = []
        orig_ref_imgs = []
        for instance in instances:
            clip_images.append(instance["clip_image"])
            sam_images.append(instance["sam_image"])
            sam_resized_sizes.append(instance["sam_resized_size"])

            gt_classes.append(instance["gt_classes"])
            gt_masks.append(instance["gt_masks"])
            # gt_bboxes.append(instance["gt_bbox"])

            # print(instance["conversation"])
            conversations.append(instance["conversation"])
            questions.append(instance["question"])

            file_names.append(instance["file_name"])
            heights.append(instance["height"])
            widths.append(instance["width"])

            audios.append(None)

            if "ref_image" in instance:
                ref_images.append(instance["ref_image"])
            else:
                ref_images.append(None)
                # pass

            orig_ref_imgs.append(instance["ref_img"])

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
            image_id=image_ids,
            audios=audios,
            ref_images=ref_images,
            orig_ref_imgs=orig_ref_imgs,
        )


class mycoco20i(COCOSemantic):
    def __init__(
        self,
        split: int = 0,
        shot: int = 1,
        mode: str = "val",
        is_train: bool = True,
        image_root: str = "/mnt/data/data/coco/train2017/",
        json_file: str = "/mnt/data/data/coco/annotations/instances_train2017.json",
        apply_mask: bool = False,
    ):
        super().__init__(is_train, image_root, json_file, num_sample=0)

        fss_list_root = f"/mnt/data/data/coco_20i/lists/coco/fss_list/{mode}/"
        fss_data_list_path = fss_list_root + "data_list_{}.txt".format(split)
        with open(fss_data_list_path, "r") as f:
            f_str = f.readlines()
        self.data_list = []
        for line in f_str:
            img, mask = line.split(" ")
            self.data_list.append((img, mask.strip()))

        self.coco_id_list = coco20_split[split]

        self.apply_mask = apply_mask

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image_id = int(os.path.basename(image_path).split(".")[0][-12:])
        # print(image_id)

        try:
            index = self.ids.index(image_id)
        except:  # noqa: E722
            print("ERROR: ", image_id)
            # return None
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))

        data_dict = super().__getitem__(index)

        gt_classes = data_dict["gt_classes"].tolist()
        for c in gt_classes:
            if c in self.coco_id_list:
                gt_class = c
        gt_idx = gt_classes.index(gt_class)
        gt_mask = data_dict["gt_masks"][gt_idx]

        # if mask area / image area < 0.1, return None
        if gt_mask.sum() / (gt_mask.shape[0] * gt_mask.shape[1]) < 0.1:
            # return None
            print("mask too small")
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))

        data_dict["gt_classes"] = torch.tensor(gt_class)
        data_dict["gt_masks"] = gt_mask

        refs = self.samples[str(gt_class)]
        # print(len(refs))
        ref_idx = random.choice(refs)
        # print(ref_idx)
        ref_data_dict = super().__getitem__(ref_idx[0])
        try:
            if ref_data_dict["gt_classes"][ref_idx[1]] != gt_class:
                # print(ref_data_dict["gt_classes"][ref_idx[1]], gt_class)
                print("class error")
                # return None
                return self.__getitem__(random.randint(0, len(self.data_list) - 1))
        except:  # noqa: E722
            # return None
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))
        ref_img = cv2.imread(ref_data_dict["file_name"])
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

        if self.apply_mask:
            ref_gt_mask = ref_data_dict["gt_masks"][ref_idx[1]]
            # print(ref_img.shape, ref_gt_mask.shape)

            # data_dict["ref_gt_mask"] = ref_gt_mask
            # data_dict["orig_ref_img"] = ref_img
            ref_img = ref_img * ref_gt_mask.unsqueeze(-1).numpy()

        x1, y1, w, h = ref_data_dict["bbox"][ref_idx[1]].int().tolist()
        ref_img = ref_img[y1 + 1 : y1 + h - 1, x1 + 1 : x1 + w - 1]

        data_dict["ref_img"] = ref_img

        return data_dict


class mycoco20i_Tokenized(mycoco20i):
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
        split: int = 0,
        mode: str = "val",
        apply_mask: bool = False,
    ):
        super().__init__(split=split, mode=mode, apply_mask=apply_mask)

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

        # print(gt_classes)
        try:
            gt_class = thing_dataset_id_to_name[gt_classes.item()]
        except:  # noqa: E722
            gt_class = gt_classes[0]
        # gt_class = thing_dataset_id_to_name[gt_classes.item()]

        # gt_class = gt_classes[0]
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
        if gt_masks.dim() == 2:
            dataset_dict["gt_masks"] = gt_masks.unsqueeze(0)
        else:
            dataset_dict["gt_masks"] = gt_masks
        # dataset_dict["gt_bbox"] = gt_bbox

        dataset_dict["ref_image"] = ref_image
        # print(conversation)

        return dataset_dict


# class SemData(Dataset):
#     def __init__(
#         self,
#         split=3,
#         shot=1,
#         data_root=None,
#         base_data_root=None,
#         data_list=None,
#         data_set=None,
#         use_split_coco=False,
#         transform=None,
#         transform_tri=None,
#         mode="train",
#         ann_type="mask",
#         ft_transform=None,
#         ft_aug_size=None,
#         ms_transform=None,
#     ):
#         assert mode in ["train", "val", "demo", "finetune"]
#         assert data_set in ["pascal", "coco"]
#         if mode == "finetune":
#             assert ft_transform is not None
#             assert ft_aug_size is not None

#         if data_set == "pascal":
#             self.num_classes = 20
#         elif data_set == "coco":
#             self.num_classes = 80

#         self.mode = mode
#         self.split = split
#         self.shot = shot
#         self.data_root = data_root
#         self.base_data_root = base_data_root
#         self.ann_type = ann_type

#         if data_set == "pascal":
#             self.class_list = list(
#                 range(1, 21)
#             )  # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
#             if self.split == 3:
#                 self.sub_list = list(
#                     range(1, 16)
#                 )  # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#                 self.sub_val_list = list(range(16, 21))  # [16,17,18,19,20]
#             elif self.split == 2:
#                 self.sub_list = list(range(1, 11)) + list(
#                     range(16, 21)
#                 )  # [1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
#                 self.sub_val_list = list(range(11, 16))  # [11,12,13,14,15]
#             elif self.split == 1:
#                 self.sub_list = list(range(1, 6)) + list(
#                     range(11, 21)
#                 )  # [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
#                 self.sub_val_list = list(range(6, 11))  # [6,7,8,9,10]
#             elif self.split == 0:
#                 self.sub_list = list(
#                     range(6, 21)
#                 )  # [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
#                 self.sub_val_list = list(range(1, 6))  # [1,2,3,4,5]

#         elif data_set == "coco":
#             if use_split_coco:
#                 print("INFO: using SPLIT COCO (FWB)")
#                 self.class_list = list(range(1, 81))
#                 if self.split == 3:
#                     self.sub_val_list = list(range(4, 81, 4))
#                     self.sub_list = list(set(self.class_list) - set(self.sub_val_list))
#                 elif self.split == 2:
#                     self.sub_val_list = list(range(3, 80, 4))
#                     self.sub_list = list(set(self.class_list) - set(self.sub_val_list))
#                 elif self.split == 1:
#                     self.sub_val_list = list(range(2, 79, 4))
#                     self.sub_list = list(set(self.class_list) - set(self.sub_val_list))
#                 elif self.split == 0:
#                     self.sub_val_list = list(range(1, 78, 4))
#                     self.sub_list = list(set(self.class_list) - set(self.sub_val_list))
#             else:
#                 print("INFO: using COCO (PANet)")
#                 self.class_list = list(range(1, 81))
#                 if self.split == 3:
#                     self.sub_list = list(range(1, 61))
#                     self.sub_val_list = list(range(61, 81))
#                 elif self.split == 2:
#                     self.sub_list = list(range(1, 41)) + list(range(61, 81))
#                     self.sub_val_list = list(range(41, 61))
#                 elif self.split == 1:
#                     self.sub_list = list(range(1, 21)) + list(range(41, 81))
#                     self.sub_val_list = list(range(21, 41))
#                 elif self.split == 0:
#                     self.sub_list = list(range(21, 81))
#                     self.sub_val_list = list(range(1, 21))

#         print("sub_list: ", self.sub_list)
#         print("sub_val_list: ", self.sub_val_list)

#         # @@@ For convenience, we skip the step of building datasets and instead use the pre-generated lists @@@
#         # if self.mode == 'train':
#         #     self.data_list, self.sub_class_file_list = make_dataset(split, data_root, data_list, self.sub_list, True)
#         #     assert len(self.sub_class_file_list.keys()) == len(self.sub_list)
#         # elif self.mode == 'val' or self.mode == 'demo' or self.mode == 'finetune':
#         #     self.data_list, self.sub_class_file_list = make_dataset(split, data_root, data_list, self.sub_val_list, False)
#         #     assert len(self.sub_class_file_list.keys()) == len(self.sub_val_list)

#         mode = "train" if self.mode == "train" else "val"
#         self.base_path = os.path.join(self.base_data_root, mode, str(self.split))

#         # fss_list_root = './lists/{}/fss_list/{}/'.format(data_set, mode)
#         fss_list_root = "/mnt/data/data/coco_20i/lists/{}/fss_list/{}/".format(
#             data_set, mode
#         )
#         fss_data_list_path = fss_list_root + "data_list_{}.txt".format(split)
#         fss_sub_class_file_list_path = (
#             fss_list_root + "sub_class_file_list_{}.txt".format(split)
#         )
#         print(fss_data_list_path)
#         print(fss_sub_class_file_list_path)

#         # Write FSS Data
#         # with open(fss_data_list_path, 'w') as f:
#         #     for item in self.data_list:
#         #         img, label = item
#         #         f.write(img + ' ')
#         #         f.write(label + '\n')
#         # with open(fss_sub_class_file_list_path, 'w') as f:
#         #     f.write(str(self.sub_class_file_list))

#         # Read FSS Data
#         with open(fss_data_list_path, "r") as f:
#             f_str = f.readlines()
#         self.data_list = []
#         for line in f_str:
#             img, mask = line.split(" ")
#             self.data_list.append((img, mask.strip()))

#         with open(fss_sub_class_file_list_path, "r") as f:
#             f_str = f.read()
#         # print(f_str)
#         self.sub_class_file_list = eval(f_str)

#         self.transform = transform
#         self.transform_tri = transform_tri
#         self.ft_transform = ft_transform
#         self.ft_aug_size = ft_aug_size
#         self.ms_transform_list = ms_transform

#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, index):
#         label_class = []
#         image_path, label_path = self.data_list[index]
#         # if "train" in image_path:
#         #     sub_dir = "train2017"
#         # else:
#         #     sub_dir = "val2017"
#         # print(image_path, label_path)
#         # print(os.path.basename(image_path))
#         id = os.path.basename(image_path).split(".")[0][-12:]
#         image_path = os.path.join(COCO_BASE, "train2017", f"{id}.jpg")
#         if not os.path.exists(image_path):
#             image_path = image_path.replace("train2017", "val2017")
#         # print(image_path)

#         label_basename = os.path.basename(label_path)
#         # if "train" in label_path:
#         #     sub_dir = "train"
#         # else:
#         #     sub_dir = "val"
#         label_path = os.path.join(
#             COCO20i_BASE, "coco", "train", str(self.split), label_basename
#         )
#         if not os.path.exists(label_path):
#             label_path = label_path.replace("train", "val")
#         # print(image_path, label_path)

#         image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = np.float32(image)
#         label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
#         label_b = cv2.imread(
#             os.path.join(self.base_path, label_path.split("/")[-1]),
#             cv2.IMREAD_GRAYSCALE,
#         )

#         if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
#             raise (
#                 RuntimeError(
#                     "Query Image & label shape mismatch: "
#                     + image_path
#                     + " "
#                     + label_path
#                     + "\n"
#                 )
#             )
#         label_class = np.unique(label).tolist()
#         # print(label_class)
#         if 0 in label_class:
#             label_class.remove(0)
#         if 255 in label_class:
#             label_class.remove(255)
#         new_label_class = []
#         for c in label_class:
#             if c in self.sub_val_list:
#                 if self.mode == "val" or self.mode == "demo" or self.mode == "finetune":
#                     new_label_class.append(c)
#             if c in self.sub_list:
#                 if self.mode == "train":
#                     new_label_class.append(c)
#         label_class = new_label_class
#         print(label_class)
#         assert len(label_class) > 0

#         class_chosen = label_class[random.randint(1, len(label_class)) - 1]
#         while class_chosen not in self.sub_class_file_list:
#             class_chosen = label_class[random.randint(1, len(label_class)) - 1]
#         print(class_chosen)
#         target_pix = np.where(label == class_chosen)
#         ignore_pix = np.where(label == 255)
#         label[:, :] = 0
#         if target_pix[0].shape[0] > 0:
#             label[target_pix[0], target_pix[1]] = 1
#         label[ignore_pix[0], ignore_pix[1]] = 255

#         # for cls in range(1,self.num_classes+1):
#         #     select_pix = np.where(label_b_tmp == cls)
#         #     if cls in self.sub_list:
#         #         label_b[select_pix[0],select_pix[1]] = self.sub_list.index(cls) + 1
#         #     else:
#         #         label_b[select_pix[0],select_pix[1]] = 0

#         file_class_chosen = self.sub_class_file_list[class_chosen]
#         num_file = len(file_class_chosen)
#         print("num_file: ", num_file)

#         support_image_path_list = []
#         support_label_path_list = []
#         support_idx_list = []
#         for k in range(self.shot):
#             support_idx = random.randint(1, num_file) - 1
#             support_image_path = image_path
#             support_label_path = label_path
#             while (
#                 support_image_path == image_path and support_label_path == label_path
#             ) or support_idx in support_idx_list:
#                 support_idx = random.randint(1, num_file) - 1
#                 support_image_path, support_label_path = file_class_chosen[support_idx]
#             support_idx_list.append(support_idx)
#             support_image_path_list.append(support_image_path)
#             support_label_path_list.append(support_label_path)

#         support_image_list_ori = []
#         support_label_list_ori = []
#         support_label_list_ori_mask = []
#         subcls_list = []
#         if self.mode == "train":
#             subcls_list.append(self.sub_list.index(class_chosen))
#         else:
#             subcls_list.append(self.sub_val_list.index(class_chosen))
#         for k in range(self.shot):
#             support_image_path = support_image_path_list[k]
#             support_label_path = support_label_path_list[k]
#             # print(support_image_path, support_label_path)
#             id = os.path.basename(support_image_path).split(".")[0][-12:]
#             support_image_path = os.path.join(COCO_BASE, "train2017", f"{id}.jpg")
#             if not os.path.exists(support_image_path):
#                 support_image_path = support_image_path.replace("train2017", "val2017")

#             label_basename = os.path.basename(support_label_path)
#             support_label_path = os.path.join(
#                 COCO20i_BASE, "coco", "train", str(self.split), label_basename
#             )
#             if not os.path.exists(support_label_path):
#                 support_label_path = support_label_path.replace("train", "val")
#             # print(label_path)
#             print(support_image_path, support_label_path)

#             support_image = cv2.imread(support_image_path, cv2.IMREAD_COLOR)
#             support_image = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)
#             support_image = np.float32(support_image)
#             support_label = cv2.imread(support_label_path, cv2.IMREAD_GRAYSCALE)
#             print("support label: ", np.unique(support_label))
#             target_pix = np.where(support_label == class_chosen)
#             ignore_pix = np.where(support_label == 255)
#             support_label[:, :] = 0
#             support_label[target_pix[0], target_pix[1]] = 1
#             print(support_label.shape)
#             print(ignore_pix)

#             # support_label, support_label_mask = transform_anns(support_label, self.ann_type)   # mask/bbox
#             support_label[ignore_pix[0], ignore_pix[1]] = 255
#             support_label_mask[ignore_pix[0], ignore_pix[1]] = 255
#             if (
#                 support_image.shape[0] != support_label.shape[0]
#                 or support_image.shape[1] != support_label.shape[1]
#             ):
#                 raise (
#                     RuntimeError(
#                         "Support Image & label shape mismatch: "
#                         + support_image_path
#                         + " "
#                         + support_label_path
#                         + "\n"
#                     )
#                 )
#             support_image_list_ori.append(support_image)
#             support_label_list_ori.append(support_label)
#             support_label_list_ori_mask.append(support_label_mask)
#         assert (
#             len(support_label_list_ori) == self.shot
#             and len(support_image_list_ori) == self.shot
#         )

#         raw_image = image.copy()
#         raw_label = label.copy()
#         raw_label_b = label_b.copy()
#         support_image_list = [[] for _ in range(self.shot)]
#         support_label_list = [[] for _ in range(self.shot)]
#         if self.transform is not None:
#             image, label, label_b = self.transform_tri(
#                 image, label, label_b
#             )  # transform the triple
#             for k in range(self.shot):
#                 support_image_list[k], support_label_list[k] = self.transform(
#                     support_image_list_ori[k], support_label_list_ori[k]
#                 )

#         s_xs = support_image_list
#         s_ys = support_label_list
#         s_x = s_xs[0].unsqueeze(0)
#         for i in range(1, self.shot):
#             s_x = torch.cat([s_xs[i].unsqueeze(0), s_x], 0)
#         s_y = s_ys[0].unsqueeze(0)
#         for i in range(1, self.shot):
#             s_y = torch.cat([s_ys[i].unsqueeze(0), s_y], 0)

#         # Return
#         if self.mode == "train":
#             return image, label, label_b, s_x, s_y, subcls_list
#         elif self.mode == "val":
#             return image, label, label_b, s_x, s_y, subcls_list, raw_label, raw_label_b
#         elif self.mode == "demo":
#             total_image_list = support_image_list_ori.copy()
#             total_image_list.append(raw_image)
#             return (
#                 image,
#                 label,
#                 label_b,
#                 s_x,
#                 s_y,
#                 subcls_list,
#                 total_image_list,
#                 support_label_list_ori,
#                 support_label_list_ori_mask,
#                 raw_label,
#                 raw_label_b,
#             )
