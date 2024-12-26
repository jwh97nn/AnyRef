from dataclasses import dataclass, field

import torch
import transformers
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer, Trainer

from model.anyref import AnyRefForCausalLM
from model.llava.constants import (
    AUDIO_REF_END_TOKEN,
    AUDIO_REF_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMG_REF_END_TOKEN,
    IMG_REF_START_TOKEN,
)
from utils.ade_instance import ADEInstanceTokenized
from utils.ade_semantic import ADESemanticTokenized
from utils.avsbench import AVSMultiTokenized, AVSObjectTokenized
from utils.coco_instance import (
    COCOInstanceTokenized,
    COCOSemantic_inv_Tokenized,
    DataCollector,
)
from utils.reason import ReasonSegTokenized
from utils.refer_seg import REFCOCOTokenized
from utils.refer_seg_invert import REFCOCO_invTokenized


@dataclass()
class ModelArguments:
    add_audio_encoder: str = field(default=False)
    rephrase_weight: float = field(default=0.0)
    roi: bool = field(default=False)


@dataclass()
class DataArguments:
    train_datasets: str = field(default="")
    clip_resize_wo_crop: bool = field(default=True)
    itisseg: bool = field(default=False)
    no_mask: bool = field(default=False)


@dataclass()
class TrainingArguments(transformers.TrainingArguments):
    lora_name: str = field(default="")


def train():
    # global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(
        "LLaVA-Lightning-7B-v1-1",
        padding_side="right",
        use_fast=False,
        model_max_length=512,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # Tag: multiple obj tokens (do not work!)
    num_obj_token = 1
    seg_token_idx = []

    # Tag: add loc tokens
    # with_bbox = True
    with_bbox = False
    loc_token_idx = []

    # Tag: add <seg_start>, <seg_end> tokens
    # seg_start_end = False   # add this will make tokenizer doing wrong

    # Tag: add audio tokens
    # add_audio_encoder = True
    add_audio_encoder = model_args.add_audio_encoder

    # Tag: add img_ref tokens
    add_img_ref = True
    # add_img_ref = False

    # Tag: multi modality
    # multi_modality = True
    multi_modality = False

    clip_resize_wo_crop = data_args.clip_resize_wo_crop
    # train_dataset = "coco_instance"
    # train_dataset = "ade_instance"
    # train_dataset = "ade_semantic"
    # train_dataset = "refer_seg,ade_semantic,refer_seg_inv"
    train_dataset = data_args.train_datasets

    itisseg = data_args.itisseg
    # itisseg = True  # answer = "it is [SEG]."

    if num_obj_token == 1:
        if with_bbox:
            for i in range(101):
                tokenizer.add_tokens(f"[LOC{i}]")
                loc_token_idx.append(
                    tokenizer(f"[LOC{i}]", add_special_tokens=False).input_ids[0]
                )
        else:
            loc_token_idx = None
        tokenizer.add_tokens("[SEG]")
        seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    else:
        for i in range(num_obj_token):
            tokenizer.add_tokens(f"[SEG{i}]")
            seg_token_idx.append(
                tokenizer(f"[SEG{i}]", add_special_tokens=False).input_ids[0]
            )
    tokenizer.add_tokens(
        [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
    )
    # if add_audio_encoder:
    tokenizer.add_tokens(
        [AUDIO_REF_START_TOKEN, AUDIO_REF_END_TOKEN], special_tokens=True
    )

    # if seg_start_end:
    #     tokenizer.add_tokens(
    #         [SEG_START_TOKEN, SEG_END_TOKEN], special_tokens=True
    #     )

    # if add_img_ref:
    tokenizer.add_tokens([IMG_REF_START_TOKEN, IMG_REF_END_TOKEN], special_tokens=True)

    sampled_class_num = 1
    # sampled_class_num = 3

    dataset_list = []
    for dataset in train_dataset.split(","):
        if dataset == "coco_instance":
            dataset_list.append(
                COCOInstanceTokenized(
                    tokenizer=tokenizer,
                    filter_area=0.01,  # Tag: filter area threshold
                    with_bbox=with_bbox,
                    sampled_class_num=sampled_class_num,
                    clip_resize_wo_crop=clip_resize_wo_crop,
                )
            )
        elif dataset == "coco_inv":
            dataset_list.append(
                COCOSemantic_inv_Tokenized(
                    tokenizer=tokenizer,
                    img_ref=True,
                    placehold=True,
                    clip_resize_wo_crop=clip_resize_wo_crop,
                    # filter_area=0.05,
                    # apply_mask=0.5,
                )
            )
        elif dataset == "ade_instance":
            dataset_list.append(
                ADEInstanceTokenized(
                    tokenizer=tokenizer,
                    with_bbox=with_bbox,
                    filter_area=0.01,
                    sampled_class_num=sampled_class_num,
                    clip_resize_wo_crop=clip_resize_wo_crop,
                )
            )
        elif dataset == "ade_semantic":
            dataset_list.append(
                ADESemanticTokenized(
                    tokenizer=tokenizer,
                    with_bbox=with_bbox,
                    filter_area=0.01,
                    sampled_class_num=sampled_class_num,
                    clip_resize_wo_crop=clip_resize_wo_crop,
                    itisseg=itisseg,
                )
            )
        elif dataset == "grefcoco":
            dataset_list.append(
                REFCOCOTokenized(
                    tokenizer=tokenizer,
                    datasets="grefcoco",
                    sampled_class_num=sampled_class_num,
                    clip_resize_wo_crop=clip_resize_wo_crop,
                    # seg_start_end=seg_start_end,
                    itisseg=itisseg,
                )
            )
        elif dataset == "refer_seg":
            dataset_list.append(
                REFCOCOTokenized(
                    tokenizer=tokenizer,
                    datasets="refcoco,refcoco+,refcocog,refclef",
                    sampled_class_num=sampled_class_num,
                    clip_resize_wo_crop=clip_resize_wo_crop,
                    itisseg=itisseg,
                )
            )
        elif dataset == "refer_seg_refcoco+":
            dataset_list.append(
                REFCOCOTokenized(
                    tokenizer=tokenizer,
                    datasets="refcoco+",
                    sampled_class_num=sampled_class_num,
                    clip_resize_wo_crop=clip_resize_wo_crop,
                    itisseg=itisseg,
                )
            )
        elif dataset == "refer_seg_inv":
            dataset_list.append(
                REFCOCO_invTokenized(
                    tokenizer=tokenizer,
                    datasets="refcoco,refcoco+,refcocog,refclef",
                    clip_resize_wo_crop=clip_resize_wo_crop,
                    img_ref=add_img_ref,
                    placehold=True,
                )
            )
        elif dataset == "refer_seg_inv_refcocog":
            dataset_list.append(
                REFCOCO_invTokenized(
                    tokenizer=tokenizer,
                    datasets="refcocog",
                    clip_resize_wo_crop=clip_resize_wo_crop,
                    img_ref=add_img_ref,
                    placehold=True,
                    split="",
                )
            )
        elif dataset == "refer_seg_inv_refcoco":
            dataset_list.append(
                REFCOCO_invTokenized(
                    tokenizer=tokenizer,
                    datasets="refcoco",
                    clip_resize_wo_crop=clip_resize_wo_crop,
                    img_ref=add_img_ref,
                    placehold=True,
                    split="",
                )
            )
        elif dataset == "refer_seg_inv_refcoco+":
            dataset_list.append(
                REFCOCO_invTokenized(
                    tokenizer=tokenizer,
                    datasets="refcoco+",
                    clip_resize_wo_crop=clip_resize_wo_crop,
                    img_ref=add_img_ref,
                    placehold=True,
                    split="",
                )
            )
        elif dataset == "reason":
            dataset_list.append(
                ReasonSegTokenized(
                    tokenizer=tokenizer,
                    split="train",
                    clip_resize_wo_crop=clip_resize_wo_crop,
                    itisseg=True,
                )
            )
        elif dataset == "avs_object":
            dataset_list.append(
                AVSObjectTokenized(
                    split="train",
                    tokenizer=tokenizer,
                    clip_resize_wo_crop=clip_resize_wo_crop,
                    itisseg=itisseg,
                    # itisseg=False,
                    multi_modality=multi_modality,
                    # seg_start_end=seg_start_end,
                    placehold=True,
                )
            )
        elif dataset == "avs_multi":
            dataset_list.append(
                AVSMultiTokenized(
                    split="train",
                    tokenizer=tokenizer,
                    clip_resize_wo_crop=clip_resize_wo_crop,
                    # original_resolution=True,
                    itisseg=itisseg,
                    multi_modality=multi_modality,
                    # seg_start_end=seg_start_end,
                    placehold=True,
                )
            )
        else:
            raise NotImplementedError

    if len(dataset_list) == 1:
        train_dataset = dataset_list[0]
    else:
        train_dataset = torch.utils.data.ConcatDataset(dataset_list)

    data_collator = DataCollector(
        tokenizer=tokenizer,
        conv_type="llava_v1",
        use_mm_start_end=True,
        roi=model_args.roi,
    )

    torch_dtype = torch.float32
    if training_args.fp16:
        torch_dtype = torch.float16

    model_args = {
        "train_mask_decoder": True,
        "out_dim": 256,
        "ce_loss_weight": 1,
        "dice_loss_weight": 0.5,
        "bce_loss_weight": 2.0,
        "seg_token_idx": seg_token_idx,
        # "vision_pretrained": "SAM/sam_vit_l_0b3195.pth",
        "vision_pretrained": "SAM/sam_vit_h_4b8939.pth",
        "vision_tower": "openai/clip-vit-large-patch14/",
        "use_mm_start_end": True,
        "loc_token_idx": loc_token_idx,  # Tag: add loc tokens
        "loc_weight": 0.1,  # Tag: loc_tokens weight
        "add_audio_encoder": add_audio_encoder,  # Tag: audio
        "rephrase_weight": model_args.rephrase_weight,
    }
    model = AnyRefForCausalLM.from_pretrained(
        "LLaVA-Lightning-7B-v1-1",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        **model_args,
    )

    model.get_model().initialize_vision_modules(model.get_model().config)  # LLaVA ViT
    model.get_vision_tower().to(torch_dtype)
    model.get_model().initialize_anyref_modules(model.get_model().config)  # SAM
    # model.get_model().visual_model.image_encoder.to(torch_dtype)  # SAM encoder to fp16

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    for p in model.get_model().get_vision_tower().parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False
    model.resize_token_embeddings(len(tokenizer))

    # ---------------- LORA ----------------
    def find_linear_layers(model, lora_target_modules):
        cls = torch.nn.Linear
        lora_module_names = set()
        for name, module in model.named_modules():
            if (
                isinstance(module, cls)
                and all(
                    [
                        x not in name
                        for x in [
                            "visual_model",
                            "vision_tower",
                            "mm_projector",
                            "text_hidden_fcs",
                        ]
                    ]
                )
                and any([x in name for x in lora_target_modules])
            ):
                lora_module_names.add(name)
        return sorted(list(lora_module_names))

    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    lora_target_modules = find_linear_layers(model, "q_proj,v_proj".split(","))
    modules_to_save = [
        "mask_decoder.mask_tokens",
        "output_upscaling",
        "output_hypernetworks_mlps",
        "embed_tokens",
        "lm_head",
        "text_hidden_fcs",
    ]
    if with_bbox:
        # modules_to_save.append("loc_fc")
        modules_to_save.append("loc_embeddings")
    if add_audio_encoder:
        modules_to_save.append("audio_projector")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=modules_to_save,
    )
    # model = get_peft_model(model, lora_config)

    # LoRA resume
    # lora_name = "output/refer_ade/checkpoint-10000"
    lora_name = training_args.lora_name
    if lora_name == "no":
        model = get_peft_model(model, lora_config)

    else:
        if lora_name == "":
            lora_name = "output/refer_ade/checkpoint-10000"

        model = PeftModel.from_pretrained(model, lora_name)
        print(f"loaded lora: {lora_name}")

    model.print_trainable_parameters()
    # ---------------- LORA ----------------

    model.get_model().embed_tokens.to(torch.float32)
    model.lm_head.to(torch.float32)
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n, p.dtype)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train(
        # "output2/refer_inv_roi_nomask/checkpoint-1900"
    )


if __name__ == "__main__":
    train()
