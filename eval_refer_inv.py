import json
import os

import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
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
from utils.coco_instance import DataCollector
from utils.refer_seg_invert import REFCOCO_val, REFCOCO_val_invTokenized


@torch.no_grad()
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
    clip_resize_wo_crop: bool = True,
    roi: bool = False,
    no_mask: bool = False,
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

    if seg_start_end:
        tokenizer.add_tokens([SEG_START_TOKEN, SEG_END_TOKEN], special_tokens=True)

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
    datacollector = DataCollector(tokenizer, roi=roi)

    for val_ds in val_datasets.split(","):
        val, split = val_ds.split("_")
        dataset = REFCOCO_val_invTokenized(
            tokenizer,
            dataset=val,
            split=split,
            seg_start_end=seg_start_end,
            img_ref=True,
            placehold=True,
            clip_resize_wo_crop=clip_resize_wo_crop,
        )
        dataloader = DataLoader(
            dataset, batch_size=1, collate_fn=datacollector, shuffle=False
        )

        # hs, ps, orig_hs = [], [], []

        print(f"evaluating {lora_name} on {val_ds}_invert")

        pred_mask_save_path = os.path.join(lora_name, val, split, "inv_pred_masks")
        if not os.path.exists(pred_mask_save_path):
            os.makedirs(pred_mask_save_path, exist_ok=True)

        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            input_ids = data["input_ids"][:, : (data["labels"] < 0).sum()].cuda()
            clip_images = data["clip_images"].cuda().half()
            sam_images = data["sam_images"].cuda().half()
            sam_resized_sizes = data["sam_resized_sizes"]
            if isinstance(data["ref_images"], list):
                ref_images = []
                for r in data["ref_images"]:
                    if r is not None:
                        ref_images.append(r.cuda().half())
                    else:
                        ref_images.append(None)
            else:
                ref_images = data["ref_images"].cuda().half()
            # output_ids, pred_masks, (h, p, orig_h) = model.generate(
            output_ids, pred_masks = model.generate(
                clip_images,
                input_ids,
                sam_images,
                sam_resized_sizes,
                data["height"],
                data["width"],
                ref_images=ref_images,
            )

            no_mask = False
            if pred_masks is None:
                no_mask = True

            if not no_mask:
                pred_mask = pred_masks[0]  # [1, h, w]

            output_ids = output_ids[0][output_ids[0] > 0]

            text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
            text_output = text_output.split("ASSISTANT:")[1]

            # save captions to txt
            with open(os.path.join(lora_name, val, split, "exp_gen.txt"), "a") as f:
                f.write(text_output + "\n")

            if not no_mask:
                try:
                    pred_mask_save = (
                        torch.sigmoid(pred_mask[0].cpu()) > 0.5
                    ).int()  # [h, w]
                except:  # noqa: E722
                    continue
                    # pred_mask_save = (torch.zeros(pred_mask.shape[-2:])).int()
                pred_mask_save = pred_mask_save.numpy().astype(np.uint8) * 255
                pred_mask_save = Image.fromarray(pred_mask_save).convert("P")
                output_name = f"{str(i).zfill(4)}.png"
                pred_mask_save.save(
                    os.path.join(pred_mask_save_path, output_name), format="PNG"
                )

        # hs = torch.cat(hs, dim=0)   # [N, c]
        # ps = torch.cat(ps, dim=0)   # [N, c]
        # orig_hs = torch.cat(orig_hs, dim=0)   # [N, c]
        # torch.save(hs, os.path.join(lora_name, val, split, "inv_hs.pt"))
        # torch.save(ps, os.path.join(lora_name, val, split, "inv_ps.pt"))
        # torch.save(orig_hs, os.path.join(lora_name, val, split, "inv_orig_hs.pt"))

        # evaluate
        refcoco_inv = REFCOCO_val(dataset=val, split=split)
        with open(os.path.join(lora_name, val, split, "exp_gen.txt"), "r") as f:
            preds = f.readlines()
        anns, images = [], []
        image_id = 0
        for i in tqdm(range(len(refcoco_inv))):
            # ref_id = refcoco_inv.ref_ids[i]
            # ref = refcoco_inv.refer_api.loadRefs(ref_id)[0]
            images.append({"id": image_id})
            anns.append(
                {
                    "image_id": image_id,
                    "caption": preds[i].split("[SEG]")[0].strip(),
                }
            )
            image_id += 1
        out_file = os.path.join(lora_name, val, split, "converted_single2.txt")
        with open(out_file, "w") as f:
            json.dump(anns, f, indent=4)

        annotation_file = f"{val}_{split}_inv_gt.json"
        coco = COCO(annotation_file)
        coco_result = coco.loadRes(out_file)
        coco_eval = COCOEvalCap(coco, coco_result)
        coco_eval.params["image_id"] = coco_result.getImgIds()
        res = coco_eval.evaluate()
        with open(os.path.join(lora_name, val, split, "inv_result.txt"), "w") as f:
            json.dump(res, f, indent=4)

        print(f"evaluated {lora_name} on {val_ds}_invert")
        print(res)


if __name__ == "__main__":
    lora_name = "output2/refcoco_inv_roi_nomask/checkpoint-100"

    # val_datasets = "refcoco+_testA,refcoco+_testB"
    # val_datasets = "refcocog_val"
    val_datasets = "refcoco_testA, refcoco_testB"
    # val_datasets = "refcoco_testA"
    # val_datasets = "refcoco_testB"
    # val_datasets = "refcoco+_testA, refcoco+_testB"

    # val_datasets = "refcocog_val,refcoco_testA,refcoco_testB,refcoco+_testA,refcoco+_testB"
    # val_datasets = "refcocog_val,refcoco_testA"
    # val_datasets = "refcoco_testB,refcoco+_testA,refcoco+_testB"

    val_datasets = val_datasets.replace(" ", "")
    main(
        lora_name=lora_name,
        val_datasets=val_datasets,
        itisseg=False,
        # multi_modality=False,
        seg_start_end=False,
        # rephrase_weight=0.1,
        rephrase_weight=0.0,
        # clip_resize_wo_crop=False,
        roi=True,
        # roi=False,
        no_mask=True,
    )
    # print(f"evaluated {lora_name} on {val_datasets}")
