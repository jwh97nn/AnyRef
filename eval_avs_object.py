import json
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
    SEG_END_TOKEN,
    SEG_START_TOKEN,
)
from utils.avsbench import AVSMultiTokenized, AVSObjectTokenized
from utils.coco_instance import DataCollector
from utils.pyutils import AverageMeter, Eval_Fmeasure, mask_iou


def main(
    lora_name,
    dataset,
    convert_classname: bool = True,
    original_resolution: bool = True,
    itisseg: bool = False,
    multi_modality: bool = False,
    seg_start_end: bool = False,
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
    tokenizer.add_tokens(
        [AUDIO_REF_START_TOKEN, AUDIO_REF_END_TOKEN], special_tokens=True
    )
    tokenizer.add_tokens([IMG_REF_START_TOKEN, IMG_REF_END_TOKEN], special_tokens=True)

    if seg_start_end:
        tokenizer.add_tokens([SEG_START_TOKEN, SEG_END_TOKEN], special_tokens=True)

    # Model
    model_version = "LLaVA-Lightning-7B-v1-1"
    model_args = {
        "train_mask_decoder": True,
        "out_dim": 256,
        "seg_token_idx": seg_token_idx,
        "vision_pretrained": "SAM/sam_vit_h_4b8939.pth",
        "add_audio_encoder": True,
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
    convert_classname = True
    original_resolution = True
    datacollector = DataCollector(tokenizer)

    for val_ds in dataset.split(","):
        if val_ds == "object":
            avs = AVSObjectTokenized(
                tokenizer,
                split="test",
                overfit=False,
                convert_classname=convert_classname,
                original_resolution=original_resolution,
                itisseg=itisseg,
                multi_modality=multi_modality,
                seg_start_end=seg_start_end,
                placehold=True,
            )
        elif val_ds == "multi":
            avs = AVSMultiTokenized(
                tokenizer,
                split="test",
                overfit=False,
                convert_classname=convert_classname,
                original_resolution=original_resolution,
                itisseg=itisseg,
                multi_modality=multi_modality,
                seg_start_end=seg_start_end,
            )
        dataloader = DataLoader(
            avs, batch_size=1, collate_fn=datacollector, shuffle=False
        )

        print(f"evaluating {lora_name} on {val_ds}")

        pred_mask_save_path = os.path.join(lora_name, val_ds, "pred_masks")
        if not os.path.exists(pred_mask_save_path):
            os.makedirs(pred_mask_save_path, exist_ok=True)

        avg_meter_miou = AverageMeter("miou")
        avg_meter_F = AverageMeter("F_score")

        fscore_pred_mask, fscore_gt_mask = [], []

        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            input_ids = data["input_ids"][:, : (data["labels"] < 0).sum()].cuda()
            clip_images = data["clip_images"].cuda().half()
            sam_images = data["sam_images"].cuda().half()
            sam_resized_sizes = data["sam_resized_sizes"]
            audios = []
            for audio in data["audios"]:
                audios.append(audio.cuda().half())
            # audios = data["audios"].cuda().half()
            _, pred_masks, _ = model.generate(
                clip_images,
                input_ids,
                sam_images,
                sam_resized_sizes,
                data["height"],
                data["width"],
                audios=audios,
            )

            pred_mask = pred_masks[0].cpu()  # [1, h, w]
            # print("1: ", pred_mask.shape)

            # save masks
            pred_mask_save = (torch.sigmoid(pred_mask[0]) > 0.5).int()  # [h, w]
            pred_mask_save = pred_mask_save.numpy().astype(np.uint8) * 255
            pred_mask_save = Image.fromarray(pred_mask_save).convert("P")
            output_name = f"{str(i).zfill(4)}.png"
            pred_mask_save.save(
                os.path.join(pred_mask_save_path, output_name), format="PNG"
            )

            gt_mask = data["gt_masks"][0]
            if pred_mask.shape[-2:] != gt_mask.shape[-2:]:
                pred_mask = torch.nn.functional.interpolate(
                    pred_mask.unsqueeze(0),
                    size=gt_mask.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            miou = mask_iou(pred_mask, gt_mask)
            avg_meter_miou.add({"miou": miou})

            fscore_pred_mask.append(pred_mask.cuda())
            fscore_gt_mask.append(gt_mask.to(pred_mask).cuda())
            if i % 5 == 4:  # 4(0-4), 9(5-9)
                fscore_pred_mask = torch.cat(fscore_pred_mask, dim=0)
                fscore_gt_mask = torch.cat(fscore_gt_mask, dim=0)
                F_score = Eval_Fmeasure(
                    fscore_pred_mask, fscore_gt_mask, pred_mask_save_path
                )
                avg_meter_F.add({"F_score": F_score})
                fscore_pred_mask, fscore_gt_mask = [], []

            # tqdm.write(str(avg_meter_miou.get("miou")))

        miou = avg_meter_miou.get("miou")
        F_score = avg_meter_F.get("F_score")

        print(f"{val_ds} miou: {miou}, F_score: {F_score}")
        print(f"evaluated {lora_name} on {val_ds}")

        res = {"miou": float(miou), "F_score": float(F_score)}
        with open(os.path.join(lora_name, val_ds, "avs_result.txt"), "w") as f:
            json.dump(res, f, indent=4)


if __name__ == "__main__":
    lora_name = "output2/avs_multi_train/checkpoint-550"

    # dataset = "object"
    dataset = "multi"

    main(
        lora_name=lora_name,
        dataset=dataset,
        itisseg=False,
        multi_modality=False,
        seg_start_end=False,
    )
