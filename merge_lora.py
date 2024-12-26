import torch
from peft import PeftModel
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


def main(lora_name):
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
    #     tokenizer.add_tokens(
    #         [SEG_START_TOKEN, SEG_END_TOKEN], special_tokens=True
    #     )

    tokenizer.add_tokens([IMG_REF_START_TOKEN, IMG_REF_END_TOKEN], special_tokens=True)

    model_version = "LLaVA-Lightning-7B-v1-1"
    model_args = {
        "train_mask_decoder": True,
        "out_dim": 256,
        "seg_token_idx": seg_token_idx,
        "vision_pretrained": "SAM/sam_vit_h_4b8939.pth",
        "add_audio_encoder": False,
        "rephrase_weight": 0.1,
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

    model = PeftModel.from_pretrained(model, lora_name)
    model = model.merge_and_unload()
    model.to(torch.float16)

    model.save_pretrained(lora_name)


if __name__ == "__main__":
    lora_name = "output2/coco_inv/checkpoint-3000"

    main(lora_name)
