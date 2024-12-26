import os

import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from model.anyref import AnyRefForCausalLM
from model.llava.constants import (
    AUDIO_REF_END_TOKEN,
    AUDIO_REF_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMG_REF_END_TOKEN,
    IMG_REF_START_TOKEN,
)
from utils.coco_instance import DataCollector
from utils.refer_seg import REFCOCO_valTokenized
from utils.utils import AverageMeter, Summary, intersectionAndUnionGPU


def main(
    lora_name,
    val_datasets,
    # convert_classname: bool = True,
    # original_resolution: bool = True,
    itisseg: bool = False,
    # multi_modality: bool = False,
    add_audio_encoder: bool = False,
    seg_start_end: bool = False,
    rephrase_weight: float = 0.0,
    clip_resize_wo_crop: bool = False,
    batch_num: int = 1,
):
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "LLaVA-Lightning-7B-v1-1", padding_side="right", use_fast=False
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[SEG]")
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    tokenizer.add_tokens(
        [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
    )
    # if add_audio_encoder:
    tokenizer.add_tokens(
        [AUDIO_REF_START_TOKEN, AUDIO_REF_END_TOKEN], special_tokens=True
    )

    # if seg_start_end:
    # tokenizer.add_tokens(
    #     [SEG_START_TOKEN, SEG_END_TOKEN], special_tokens=True
    # )

    tokenizer.add_tokens([IMG_REF_START_TOKEN, IMG_REF_END_TOKEN], special_tokens=True)

    # Model
    model_version = "LLaVA-Lightning-7B-v1-1"
    model_args = {
        "train_mask_decoder": True,
        "out_dim": 256,
        "seg_token_idx": seg_token_idx,
        "vision_pretrained": "SAM/sam_vit_h_4b8939.pth",
        "add_audio_encoder": add_audio_encoder,
        "rephrase_weight": rephrase_weight,
    }
    model = AnyRefForCausalLM.from_pretrained(
        model_version, torch_dtype=torch.float16, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.get_model().initialize_vision_modules(model.get_model().config)
    model.get_model().get_vision_tower().to(torch.float16)
    model.get_model().initialize_anyref_modules(model.get_model().config)
    model.resize_token_embeddings(len(tokenizer))

    # Lora
    # lora_name = "output/avs_object/samH_lora_ds_convertclass_origsize/checkpoint-3800"
    # config = PeftConfig.from_pretrained(lora_name)
    model = PeftModel.from_pretrained(model, lora_name)
    model = model.merge_and_unload()
    model.to(torch.float16)
    model.eval()
    model = model.cuda()

    # Data
    if batch_num == 1:
        datacollector = DataCollector(tokenizer)
    else:
        datacollector = DataCollector(tokenizer, left_pad=True)

    for val_ds in val_datasets.split(","):
        val, split = val_ds.split("_")
        dataset = REFCOCO_valTokenized(
            tokenizer, dataset=val, split=split, clip_resize_wo_crop=clip_resize_wo_crop
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_num,
            collate_fn=datacollector,
            shuffle=False,
        )

        # hs, ps, orig_hs = [], [], []
        img_id = 0

        print(f"evaluating {lora_name} on {val_ds}")

        # split = "val"
        pred_mask_save_path = os.path.join(lora_name, val, split, "pred_masks")
        if not os.path.exists(pred_mask_save_path):
            os.makedirs(pred_mask_save_path, exist_ok=True)

        intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
        union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
        acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            if batch_num == 1:
                input_ids = data["input_ids"][:, : (data["labels"] < 0).sum()].cuda()
                # attention_masks = None
            else:
                input_ids = data["input_ids"].cuda()
                # attention_masks = data["attention_masks"].cuda()

            clip_images = data["clip_images"].cuda().half()
            sam_images = data["sam_images"].cuda().half()
            sam_resized_sizes = data["sam_resized_sizes"]

            bs = clip_images.shape[0]

            output_ids, pred_masks, (hidden_states, pred_embeddings, orig_h) = (
                model.generate(
                    clip_images,
                    input_ids,
                    sam_images,
                    sam_resized_sizes,
                    data["height"],
                    data["width"],
                    # attention_masks=attention_masks,
                )
            )
            # print(hidden_states.shape, pred_embeddings.shape, orig_h.shape)

            if batch_num == 1:
                output_ids = output_ids[0][output_ids[0] > 0]
            else:
                out_ids = []
                for ii in range(len(output_ids)):
                    out_ids.append(output_ids[ii][output_ids[ii] >= 0])
                output_ids = out_ids

            # if hidden_states is not None:
            #     hs.append(hidden_states.cpu())
            #     ps.append(pred_embeddings.cpu())
            #     orig_hs.append(orig_h.cpu())
            # else:
            #     hs.append(torch.zeros_like(hs[-1]))
            #     ps.append(torch.zeros_like(ps[-1]))
            #     orig_hs.append(torch.zeros_like(orig_hs[-1]))

            if bs == 1:
                pred_mask = pred_masks[0]  # [1, h, w]
            else:
                pred_mask = [p[0] for p in pred_masks]  # [[1, h, w] * bs]

            # save masks
            for pred_mask in pred_masks:
                try:
                    pred_mask_save = (
                        torch.sigmoid(pred_mask[0].cpu()) > 0.5
                    ).int()  # [h, w]
                except:  # noqa: E722
                    img_id += 1
                    continue

                pred_mask_save = pred_mask_save.numpy().astype(np.uint8) * 255
                pred_mask_save = Image.fromarray(pred_mask_save).convert("P")
                output_name = f"{str(img_id).zfill(4)}.png"
                pred_mask_save.save(
                    os.path.join(pred_mask_save_path, output_name), format="PNG"
                )
                img_id += 1

            gt_mask = data["gt_masks"][0].int()  # [1, h, w]
            try:
                pred_mask = (torch.sigmoid(pred_mask) > 0.5).int()
            except:  # noqa: E722
                # pred_mask = (torch.zeros(gt_mask.shape)).int()
                print("gt_mask: ", gt_mask.shape)
                pred_mask = torch.zeros_like(gt_mask).int()

            intersection, union, acc_iou = 0.0, 0.0, 0.0

            intersection_i, union_i, _ = intersectionAndUnionGPU(
                pred_mask.contiguous().clone(),
                gt_mask.contiguous().cuda(),
                2,
                ignore_index=255,
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target

            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            acc_iou = acc_iou.cpu().numpy()

            intersection_meter.update(intersection)
            union_meter.update(union)
            acc_iou_meter.update(acc_iou, n=1)

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        ciou = iou_class[1]
        giou = acc_iou_meter.avg[1]

        with open(os.path.join(lora_name, val, split, "result.txt"), "a") as f:
            f.write(f"ciou: {ciou:.4f}\ngiou: {giou:.4f}\n")

        # hs = torch.cat(hs, dim=0)  # [N, c]
        # ps = torch.cat(ps, dim=0)  # [N, c]
        # orig_hs = torch.cat(orig_hs, dim=0)
        # torch.save(hs, os.path.join(lora_name, val, split, "hs.pt"))
        # torch.save(ps, os.path.join(lora_name, val, split, "ps.pt"))
        # torch.save(orig_hs, os.path.join(lora_name, val, split, "orig_hs.pt"))

        print(f"{val_ds} {split} ciou: {ciou:.4f} giou: {giou:.4f}")
        print(f"evaluated {lora_name} on {val_ds}")


if __name__ == "__main__":
    lora_name = "output2/refer_refcoco+_train/checkpoint-4500"

    # val_datasets = "refcoco_val,refcoco_testA,refcoco_testB,refcoco+_val,refcoco+_testA,refcoco+_testB,refcocog_val,refcocog_test"

    # val_datasets = "refcoco_val,refcoco_testA,refcoco_testB,refcoco+_val"
    # val_datasets = "refcoco+_testA,refcoco+_testB,refcocog_val,refcocog_test"
    # val_datasets = "refcoco+_testB,refcocog_val,refcocog_test"

    val_datasets = "refcoco+_val, refcoco+_testA, refcoco+_testB"

    val_datasets = val_datasets.replace(" ", "")
    main(
        lora_name=lora_name,
        val_datasets=val_datasets,
        itisseg=False,
        # multi_modality=False,
        seg_start_end=False,
        rephrase_weight=0.1,
        clip_resize_wo_crop=True,
        batch_num=1,
    )
    # print(f"evaluated {lora_name} on {val_datasets}")
