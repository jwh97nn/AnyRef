from tqdm import tqdm
from transformers import AutoTokenizer
from peft import PeftConfig, PeftModel
from torch.utils.data import DataLoader
from PIL import Image
import json
import torch
import os
import numpy as np

from model.llava.constants import *
from model.LISA import LISAForCausalLM
# from utils.avsbench import *
# from utils.refer_seg_invert import *
# from utils.coco_instance import DataCollector
from utils.coco20i import COCO20i_val_invTokenized, DataCollector
from utils.utils import (AverageMeter, ProgressMeter, Summary, intersectionAndUnionGPU)

# from pycocotools.coco import COCO
# from pycocoevalcap.eval import COCOEvalCap

@torch.no_grad()
def main(
    lora_name,
    val_datasets = None,
    # convert_classname: bool = True,
    # original_resolution: bool = True,
    itisseg: bool = False,
    # multi_modality: bool = False,
    add_audio_encoder: bool = False,
    seg_start_end: bool = False,
    rephrase_weight: float = 0.0,
    clip_resize_wo_crop : bool = True,
    subdir: str = "test",
):
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "LLaVA-Lightning-7B-v1-1", padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[SEG]")
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    # if add_audio_encoder:
    tokenizer.add_tokens([AUDIO_REF_START_TOKEN, AUDIO_REF_END_TOKEN], special_tokens=True)

    if seg_start_end:
        tokenizer.add_tokens(
            [SEG_START_TOKEN, SEG_END_TOKEN], special_tokens=True
        )

    tokenizer.add_tokens(
        [IMG_REF_START_TOKEN, IMG_REF_END_TOKEN], special_tokens=True
    )


    # # Model
    # lisa_version = "LLaVA-Lightning-7B-v1-1"
    # model_args = {
    #     "train_mask_decoder": True, "out_dim": 256,
    #     "seg_token_idx": seg_token_idx,
    #     "vision_pretrained": "SAM/sam_vit_h_4b8939.pth",
    #     "add_audio_encoder": add_audio_encoder,
    #     "rephrase_weight": rephrase_weight,
    # }
    # model = LISAForCausalLM.from_pretrained(lisa_version, torch_dtype=torch.float16, **model_args)
    # model.config.eos_token_id = tokenizer.eos_token_id
    # model.config.bos_token_id = tokenizer.bos_token_id
    # model.config.pad_token_id = tokenizer.pad_token_id
    # model.get_model().initialize_vision_modules(model.get_model().config)
    # model.get_model().get_vision_tower().to(torch.float16)
    # model.get_model().initialize_lisa_modules(model.get_model().config)
    # model.resize_token_embeddings(len(tokenizer))

    # # Lora
    # # lora_name = "output/avs_object/samH_lora_ds_convertclass_origsize/checkpoint-3800"
    # # config = PeftConfig.from_pretrained(lora_name)
    # model = PeftModel.from_pretrained(model, lora_name)
    # model = model.merge_and_unload()
    # model.to(torch.float16)
    # model.eval()
    # model = model.cuda()

    # Data
    datacollector = DataCollector(tokenizer)


    for split in [0, 1, 2, 3]:
        # val, split = val_ds.split("_")
        # dataset = REFCOCO_val_invTokenized(
        dataset = COCO20i_val_invTokenized(
            tokenizer, split=split, mode="val", seg_start_end=seg_start_end, img_ref=True, placehold=True,
            clip_resize_wo_crop=clip_resize_wo_crop,
        )
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=datacollector, shuffle=True)

        # hs, ps, orig_hs = [], [], []

        print(f"evaluating {lora_name} on coco_20i split_{split} {subdir}")

        pred_mask_save_path = os.path.join(lora_name, "coco", subdir, str(split))
        if not os.path.exists(pred_mask_save_path):
            os.makedirs(pred_mask_save_path, exist_ok=True)

        intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
        union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
        acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            input_ids = data["input_ids"][:, :(data["labels"] < 0).sum()].cuda()
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
            #     clip_images, input_ids, sam_images, sam_resized_sizes, data["height"], data["width"],
            #     ref_images=ref_images
            # )

            # pred_mask = pred_masks[0] # [1, h, w]

            output_ids = output_ids[0][output_ids[0] > 0]


            text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
            text_output = text_output.split("ASSISTANT:")[1]

            # save captions to txt
            with open(os.path.join(lora_name, "coco", subdir, str(split), "exp_gen.txt"), "a") as f:
                f.write(text_output + "\n")

            # try:
            #     pred_mask_save = (torch.sigmoid(pred_mask[0].cpu()) > 0.5).int()  # [h, w]
            # except:
            #     continue
            #     # pred_mask_save = (torch.zeros(pred_mask.shape[-2:])).int()
            # pred_mask_save = pred_mask_save.numpy().astype(np.uint8) * 255
            # pred_mask_save = Image.fromarray(pred_mask_save).convert('P')
            output_name = f"{str(i).zfill(4)}"
            # pred_mask_save.save(os.path.join(pred_mask_save_path, f"{output_name}.png"), format='PNG')

            # save file
            file_name = data["file_name"][0]
            img = Image.open(file_name).convert('RGB')
            img.save(os.path.join(
                pred_mask_save_path, f"{output_name}_orig.png"
            ), format='PNG')

            ref_img = data["orig_ref_imgs"][0]
            ref_img = Image.fromarray(ref_img).convert('RGB')
            ref_img.save(os.path.join(pred_mask_save_path, f"{output_name}_ref.png", format='PNG'))

            gt_mask = data["gt_masks"][0].int() # [1, h, w]
            gt_mask = gt_mask.numpy().astype(np.uint8) * 255
            gt_mask = Image.fromarray(gt_mask).convert('P')
            gt_mask.save(os.path.join(pred_mask_save_path, f"{output_name}_gt.png"), format='PNG')


            # pred_mask = (torch.sigmoid(pred_mask) > 0.5).int()
            gt_mask = data["gt_masks"][0].int()

            intersection, union, acc_iou = 0.0, 0.0, 0.0

            # intersection_i, union_i, _ = intersectionAndUnionGPU(
            #     pred_mask.contiguous().clone(), gt_mask.contiguous().cuda(), 2, ignore_index=255
            # )
            # intersection += intersection_i
            # union += union_i
            # acc_iou += intersection_i / (union_i + 1e-5)
            # acc_iou[union_i == 0] += 1.0  # no-object target

            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            acc_iou = acc_iou.cpu().numpy()

            intersection_meter.update(intersection)
            union_meter.update(union)
            acc_iou_meter.update(acc_iou, n=1)

            if i == 100:
                break

        # hs = torch.cat(hs, dim=0)   # [N, c]
        # ps = torch.cat(ps, dim=0)   # [N, c]
        # orig_hs = torch.cat(orig_hs, dim=0)   # [N, c]
        # torch.save(hs, os.path.join(lora_name, val, split, "inv_hs.pt"))
        # torch.save(ps, os.path.join(lora_name, val, split, "inv_ps.pt"))
        # torch.save(orig_hs, os.path.join(lora_name, val, split, "inv_orig_hs.pt"))

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        ciou = iou_class[1]
        giou = acc_iou_meter.avg[1]

        with open(os.path.join(lora_name, "coco", subdir, str(split), f"result.txt"), "a") as f:
            f.write(f"ciou: {ciou:.4f}\ngiou: {giou:.4f}\n")


        print(f"evaluated {lora_name} on coco_20i split_{split} {subdir}")
        # print(res)


if __name__ == "__main__":
    lora_name = "output2/coco_inv/checkpoint-1100"


    # val_datasets = val_datasets.replace(" ", "")
    main(
        lora_name=lora_name,
        # val_datasets=val_datasets,
        itisseg=False,
        # multi_modality=False,
        seg_start_end=False,
        rephrase_weight=0.1,
        # clip_resize_wo_crop=False,
        subdir="test1"
    )
    # print(f"evaluated {lora_name} on {val_datasets}")