from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h, build_sam_vit_l, build_sam_vit_b
# try:
from .ImageBind.models.imagebind_model import ModalityType
from .ImageBind.models import imagebind_model
from .llava.constants import IMG_REF_NUM
# except:
#     pass

import wandb

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    # numerator = 2 * (inputs / scale * targets).sum(-1)
    # denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    # loss = 1 - (numerator + eps) / (denominator + eps)
    # loss = loss.sum() / (num_masks + 1e-8)
    # return loss
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    
    return loss.sum() / num_masks
    


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class AnyRefMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(AnyRefMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)

            self.loc_token_idx = kwargs.get("loc_token_idx", None)
            self.add_audio_encoder = kwargs.get("add_audio_encoder", False)
            self.imagebind_ckpt = kwargs.get("imagebind_ckpt", "model/ImageBind/imagebind_huge.pth")
        else:
            self.loc_token_idx = kwargs.get("loc_token_idx", None)
            self.add_audio_encoder = kwargs.get("add_audio_encoder", False)
            self.imagebind_ckpt = kwargs.get("imagebind_ckpt", "model/ImageBind/imagebind_huge.pth")

            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_anyref_modules(self.config)

    def initialize_anyref_modules(self, config):
        # SAM
        if "vit_b" in self.vision_pretrained:
            build_sam = build_sam_vit_b
        elif "vit_l" in self.vision_pretrained:
            build_sam = build_sam_vit_l
        elif "vit_h" in self.vision_pretrained:
            build_sam = build_sam_vit_h
        else:
            raise NotImplementedError
        self.visual_model = build_sam(self.vision_pretrained)
        print("SAM loaded!")
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

        # self.loc_fc = nn.Sequential(
        #     nn.Linear(in_dim, in_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(in_dim, out_dim),
        #     nn.Dropout(0.0),
        # )
        
        if self.loc_token_idx is not None:
            # self.pe_layer = PositionEmbeddingRandom(embed_dim=256)
            self.loc_embeddings = nn.Embedding(101, out_dim)

        if self.add_audio_encoder:
            audio_encoder, audio_hidden_size = imagebind_model.imagebind_huge()
            delete_names = ["vision", "text", "depth", "thermal", "imu"]
            for name in delete_names:
                del audio_encoder.modality_preprocessors[name]
                del audio_encoder.modality_trunks[name]
                del audio_encoder.modality_postprocessors[name]
                del audio_encoder.modality_heads[name]
            imagebind_ckpt = self.imagebind_ckpt
            try:
                imagebind_ckpt = torch.load(imagebind_ckpt, map_location="cpu")
                audio_encoder.load_state_dict(imagebind_ckpt, strict=False)
                print("ImageBind audio encoder loaded!")
            except FileNotFoundError:
                print("ImageBind audio encoder ckpt not found!")
            
            self.audio_encoder = audio_encoder
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
            self.audio_encoder.eval()

            self.audio_projector = nn.Linear(audio_hidden_size, in_dim)

class AnyRefModel(AnyRefMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(AnyRefModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class AnyRefForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", 1.0)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", 0.5)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", 2.0)

        self.seg_token_idx = kwargs.get("seg_token_idx")
        if isinstance(self.seg_token_idx, list):
            self.seg_token_start = self.seg_token_idx[0]
            self.seg_token_end = self.seg_token_idx[-1]

        # self.loc_token_idx = kwargs.get("loc_token_idx", None)
        # if self.loc_token_idx is not None:
        #     self.loc_token_start = self.loc_token_idx[0]
        #     self.loc_token_end = self.loc_token_idx[-1]

        #     self.loc_weight = kwargs.get("loc_weight", 1.0)

        self.rephrase_weight = kwargs.get("rephrase_weight", 0)

        super().__init__(config)

        self.model = AnyRefModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        # return self.model_forward(**kwargs)
        return self.model_forward_new(**kwargs)
    
    def model_forward_new(
        self, 
        clip_images: torch.FloatTensor,
        sam_images: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        sam_resized_sizes: List[List[int]],
        gt_masks: Optional[List[torch.Tensor]],
        height: List[int],
        width: List[int],
        # audios: torch.FloatTensor = None,
        audios: List[torch.FloatTensor] = None,
        # ref_images: torch.FloatTensor = None,
        ref_images: List[torch.FloatTensor] = None,
        **kwargs,
    ):
        # with torch.no_grad():
        #     image_embeddings = self.model.visual_model.image_encoder(sam_images)

        # bs = image_embeddings.shape[0]
        bs = clip_images.shape[0]

        # seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        # seg_token_mask = torch.cat(
        #     [seg_token_mask, torch.zeros((seg_token_mask.shape[0], 1)).bool()],
        #     dim=1,
        # )
        # # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
        # seg_token_mask = torch.cat(
        #     [torch.zeros((seg_token_mask.shape[0], 255)).bool(), seg_token_mask],
        #     dim=1,
        # )
        no_mask = False
        if isinstance(self.seg_token_idx, list):
            seg_token_idx = torch.where((input_ids >= self.seg_token_start) & (input_ids <= self.seg_token_end))
        else:
            seg_token_idx = torch.where(input_ids == self.seg_token_idx)

        if seg_token_idx[0].shape[0] == 0:
            no_mask = True
        
        if not no_mask:
            seg_token_idx_offset_one = (seg_token_idx[0], seg_token_idx[1] - 1 + 255)

        # Tag: get input_embeddings for [loc] tokens
        if self.loc_token_idx is not None:
            loc_token_idx = torch.where((input_ids >= self.loc_token_start) & (input_ids <= self.loc_token_end))

            # loc_token_embeds = self.model.embed_tokens(input_ids[loc_token_idx].to(clip_images.device))
            loc_tokens = input_ids[loc_token_idx].to(clip_images.device)
            loc_tokens = loc_tokens - self.loc_token_start
            # try:
            loc_token_embeds = self.model.loc_embeddings(loc_tokens)    # [4*N, dim]
            # except:
            #     loc_token_embeds = self.model.loc_embeddings.modules_to_save.default(loc_tokens)
            loc_token_embeds = loc_token_embeds.reshape(-1, 4, loc_token_embeds.shape[-1])
            loc_embeddings = loc_token_embeds.mean(dim=1)   # [N, dim]
            # try:
            #     loc_embeddings = self.model.loc_fc(loc_token_embeds)   # [N, dim]
            # except:
            #     loc_embeddings = self.model.loc_fc.modules_to_save.default(loc_token_embeds)
            loc_embeddings *= self.loc_weight

        if audios is not None:
            # _, audio_feat = self.model.audio_encoder.get_audio_feature(
            #     audios, ModalityType.AUDIO
            # )   # [B, 3, 1024]
            # audio_feat = self.model.audio_projector(audio_feat) # [B, 3, dim]
            audio_feat = []
            for audio in audios:
                if audio is None:
                    audio_feat.append(None)
                else:
                    _, audio_feat_ = self.model.audio_encoder.get_audio_feature(
                        audio, ModalityType.AUDIO
                    )   # [1, 3, 1024]
                    audio_feat_ = self.model.audio_projector(audio_feat_) # [1, 3, dim]
                    audio_feat.append(audio_feat_[0])

        if ref_images is not None:
            # if ref_images.shape[0] == bs and ref_images.ndim == 4:
            #     ref_img_feat = self.encode_images(ref_images)   # [B, 256, dim]
            #     b, l, c = ref_img_feat.shape
            #     ref_img_feat = ref_img_feat.reshape(b, l//16, 16, c).mean(dim=2)   # [B, 16, dim]
            # else:
            #     raise NotImplementedError
            ref_img_feat = []
            for ref_img in ref_images:
                if ref_img is None:
                    ref_img_feat.append(None)
                else:
                    if ref_img.dim() == 1:   # roi coordinates
                        ref_img_feat.append(ref_img)
                    else:
                        ref_img_feat_ = self.encode_images(ref_img[None, ...])   # [1, 256, dim]
                        b, ll, c = ref_img_feat_.shape
                        ref_img_feat_ = ref_img_feat_.reshape(b, ll//16, 16, c).mean(dim=2)   # [1, 16, dim]
                        if ref_img_feat_.shape[1] != IMG_REF_NUM:
                            ref_img_feat_ = ref_img_feat_.reshape(b, IMG_REF_NUM, IMG_REF_NUM, c).mean(dim=2)   # [1, 4, dim]
                        ref_img_feat.append(ref_img_feat_[0])

        output = super().forward(
            images=clip_images,
            attention_mask=attention_masks,
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
            # audios=audio_feat if audios is not None else None,
            # ref_images=ref_img_feat if ref_images is not None else None,
            audios=audio_feat,
            ref_images=ref_img_feat,
            output_attentions=True,
            **kwargs,
        )

        if no_mask:
            if wandb.run:
                wandb.log({
                    "lm_loss": output.loss,
                })

            return {
                "loss": output.loss,
                "lm_loss": output.loss,
            }

        with torch.no_grad():
            image_embeddings = self.model.visual_model.image_encoder(sam_images)
        
        last_hidden_state = output.hidden_states[-1]    # [bs, seq_len, hidden_size]
        
        # Tag: Rephrase
        if self.rephrase_weight > 0:
            last_attentions = output.attentions[-1] # [bs, num_heads, seq_len, seq_len]
            rephrase_hidden_states = []
            for i in range(bs):
                rephrase_end = seg_token_idx_offset_one[1][i]
                rephrase_start = torch.where(labels[i] > 0)[0][0] - 1 + 255
                rephrase_hidden_state = last_hidden_state[i, rephrase_start:rephrase_end, :]   # [seq_len, hidden_size]

                attn = last_attentions[i].mean(0)   # [seq_len, seq_len]
                attn = attn[rephrase_end, rephrase_start:rephrase_end]
                attn = attn / attn.sum()
                
                rephrase_hidden_state = (rephrase_hidden_state * attn.unsqueeze(-1)).sum(0)   # [hidden_size]
                rephrase_hidden_states.append(rephrase_hidden_state)

        last_hidden_state = last_hidden_state[seg_token_idx_offset_one]   # [#seg_token, hidden_size]

        if self.rephrase_weight > 0:
            for i in range(bs):
                last_hidden_state[i] += rephrase_hidden_states[i] * self.rephrase_weight

        # Tag: try to make peft happy
        try:
            pred_embeddings = self.model.text_hidden_fcs[0](last_hidden_state)   # [#seg_token, dim]
        except:  # noqa: E722
            # if "modules_to_save" in self.model.text_hidden_fcs:
            pred_embeddings = self.model.text_hidden_fcs.modules_to_save.default[0](last_hidden_state)
            # else:
                # raise KeyError

        if self.loc_token_idx is not None:  # add location embeddings
            pred_embeddings = pred_embeddings + loc_embeddings

        pred_masks = []
        # for i in range(len(pred_embeddings)):
        for b in range(bs):
            b_idx = seg_token_idx[0] == b

            pred_embeddings_ = pred_embeddings[b_idx].unsqueeze(1)  # [#seg, 1, dim]

            sparse_embeddings, dense_embeddings = self.model.visual_model.prompt_encoder(
                points=None, boxes=None, masks=None,
                text_embeds=pred_embeddings_,
            )   # [#seg, 1, dim], [#seg, dim, 64, 64]
            sparse_embeddings = sparse_embeddings.to(pred_embeddings_.dtype)
            low_res_masks, _ = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[b:b+1],
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )   # [#seg, 1, h, w]
            pred_mask = self.model.visual_model.postprocess_masks(
                masks=low_res_masks,
                input_size=sam_resized_sizes[b],
                original_size=(height[b], width[b]),
            )   # [#seg, 1, h, w]
            pred_masks.append(pred_mask.squeeze(1))

        mask_ce_loss, mask_dice_loss = 0, 0
        num_masks = 0
        for b in range(bs):
            pred_mask = pred_masks[b]
            gt_mask = gt_masks[b].to(pred_mask)

            if pred_mask.shape[-2:] != gt_mask.shape[-2:]:  # for AVS
                pred_mask = F.interpolate(
                    pred_mask.unsqueeze(0), size=gt_mask.shape[-2:],
                    mode="bilinear", align_corners=False
                ).squeeze(0)

            mask_ce_loss += sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0]
            mask_dice_loss += dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0]
            num_masks += gt_mask.shape[0]
    
        mask_ce_loss = self.bce_loss_weight * mask_ce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_ce_loss + mask_dice_loss

        if wandb.run:
            wandb.log({
                "lm_loss": output.loss,
                "ce_loss": mask_ce_loss,
                "dice_loss": mask_dice_loss,
                "mask_loss": mask_loss,
            })

        return {
            "loss": output.loss + mask_loss,
            "lm_loss": output.loss,
            "ce_loss": mask_ce_loss,
            "dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }



    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        inference: bool = False,
        **kwargs,
    ):
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1

        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )
        # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask],
            dim=1,
        )

        if inference:
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

            output_hidden_states = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                torch.cuda.empty_cache()

            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None

        else:
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)

            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states

        hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        )

        seg_token_offset = seg_token_offset[offset]

        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

        multimask_output = False
        pred_masks = []
        for i in range(len(pred_embeddings)):
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks.append(pred_mask[:, 0])

        model_output = output
        gt_masks = masks_list

        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

        output = model_output.logits

        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        loss = ce_loss
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        loss += mask_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }

    @torch.no_grad()
    def generate(
        self,
        clip_images,
        input_ids,
        sam_images,
        sam_resized_sizes,
        height,
        width,
        audios=None,
        ref_images=None,
        output_hidden_states=True,
        return_dict_in_generate=True,
        max_new_tokens=128,
        attention_masks=None,
    ):
        if audios is not None:
            if isinstance(audios, list):
                audio_feat = []
                for audio in audios:
                    if audio is None:
                        audio_feat.append(None)
                    else:
                        _, audio_feat_ = self.model.audio_encoder.get_audio_feature(
                            audio, ModalityType.AUDIO
                        )
                        audio_feat_ = self.model.audio_projector(audio_feat_)
                        audio_feat.append(audio_feat_[0])
            else:
                _, audio_feat = self.model.audio_encoder.get_audio_feature(
                    audios, ModalityType.AUDIO
                )   # [B, 3, 1024]
                audio_feat = self.model.audio_projector(audio_feat) # [B, 3, dim]

        if ref_images is not None:
            if isinstance(ref_images, list):
                ref_img_feat = []
                for ref_img in ref_images:
                    if ref_img is None:
                        ref_img_feat.append(None)
                    else:
                        if ref_img.dim() == 1:   # roi coordinates
                            ref_img_feat.append(ref_img)
                        else:
                            ref_img_feat_ = self.encode_images(ref_img[None, ...])
                            ref_img_feat.append(ref_img_feat_[0])
            else:
                bs = clip_images.shape[0]
                if ref_images.shape[0] == bs and ref_images.ndim == 4:
                    ref_img_feat = self.encode_images(ref_images)   # [B, 256, dim]
                    b, ll, c = ref_img_feat.shape
                    ref_img_feat = ref_img_feat.reshape(b, ll//16, 16, c).mean(dim=2)   # [B, 16, dim]
                    if ref_img_feat.shape[1] != IMG_REF_NUM:
                        ref_img_feat = ref_img_feat.reshape(b, IMG_REF_NUM, IMG_REF_NUM, c).mean(dim=2)   # [1, 4, dim]
                else:
                    raise NotImplementedError

        outputs = super().generate(
            images=clip_images,
            input_ids=input_ids,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            max_new_tokens=max_new_tokens,
            audios=audio_feat if audios is not None else None,
            ref_images=ref_img_feat if ref_images is not None else None,
            # audios=audio_feat,
            # ref_images=ref_img_feat,
            output_attentions=self.rephrase_weight > 0,
            attention_masks=attention_masks,
        )
        # Tag: when model.eval(), hidden_states = states of the last layer
        hidden_states = outputs.hidden_states[-1]   # [bs, outputs.sequences.length+255, hidden_size]
        
        output_ids = outputs.sequences
        # print(output_ids, output_ids.shape)

        if isinstance(self.seg_token_idx, list):
            seg_token_index = torch.where((output_ids[:, 1:] >= self.seg_token_start) & (output_ids[:, 1:] <= self.seg_token_end))
        else:
            seg_token_index = torch.where(output_ids[:, 1:] == self.seg_token_idx)
        # seg_token_index_offset_one = (seg_token_index[0], seg_token_index[1] + 255)

        if seg_token_index[0].shape[0] == 0:
            return output_ids, None, (None, None, None)

        no_mask = False
        # Tag: Rephrase
        bs = input_ids.shape[0]
        if self.rephrase_weight > 0:
            # last_attentions = outputs.attentions[-1][-1] # [bs, num_heads, seq_len, seq_len]
            last_attentions = outputs.attentions[-1] # [bs, num_heads, seq_len, seq_len]
            rephrase_hidden_states = []
            for i in range(bs):
                try:
                    rephrase_end = seg_token_index[1][i] + 255
                except IndexError:
                    no_mask = True
                    break
                rephrase_start = input_ids[i, 1:].shape[0] + 255
                rephrase_hidden_state = hidden_states[i, rephrase_start:rephrase_end, :]   # [seq_len, hidden_size]

                attn = last_attentions[i].mean(0)   # [seq_len, seq_len]
                attn = attn[rephrase_end, rephrase_start:rephrase_end]
                attn = attn / attn.sum()

                # return attn
                
                rephrase_hidden_state = (rephrase_hidden_state * attn.unsqueeze(-1)).sum(0)   # [hidden_size]
                rephrase_hidden_states.append(rephrase_hidden_state)

        try:
            hidden_states = hidden_states[seg_token_index[0], seg_token_index[1] + 255, :]
            # orig_h = hidden_states.detach().clone()
        except IndexError:
            no_mask = True

        if no_mask:
            pred_masks = [torch.zeros((1, height[0], width[0])).to(clip_images)] * bs
            return output_ids, pred_masks, (None, None, None)
        
        if self.rephrase_weight > 0:
            for i in range(bs):
                hidden_states[i] += rephrase_hidden_states[i] * self.rephrase_weight
        pred_embeddings = self.model.text_hidden_fcs[0](hidden_states)  # [#seg, dim]

        # Tag: add location embeddings
        # if self.loc_token_idx is not None:
        #     # num_seg_token = pred_embeddings.shape[0]
            
        #     loc_token_idx = torch.where(
        #         (output_ids[:, 1:] >= self.loc_token_start) & (output_ids[:, 1:] <= self.loc_token_end)
        #     )
        #     # print(output_ids[:, 1:][loc_token_idx])
            
        #     # loc_token_embeds = self.model.embed_tokens(output_ids[:, 1:][loc_token_idx].to(clip_images.device))
        #     loc_tokens = output_ids[:, 1:][loc_token_idx]
        #     loc_tokens = loc_tokens - self.loc_token_start
        #     loc_token_embeds = self.model.loc_embeddings(loc_tokens)    # [4*N, dim]

        #     loc_token_embeds = loc_token_embeds.reshape(-1, 4, loc_token_embeds.shape[-1])
        #     loc_embeddings = loc_token_embeds.mean(dim=1)   # [N, dim]
        #     # loc_embeddings = self.model.loc_fc(loc_token_embeds)   # [N, dim]
        #     loc_embeddings *= self.loc_weight

        #     pred_embeddings = pred_embeddings + loc_embeddings

        image_embeddings = self.model.visual_model.image_encoder(sam_images)

        bs = image_embeddings.shape[0]
        pred_masks = []
        for b in range(bs):
            b_idx = seg_token_index[0] == b

            pred_embeddings_ = pred_embeddings[b_idx].unsqueeze(1)  # [#seg, 1, dim]

            sparse_embeddings, dense_embeddings = self.model.visual_model.prompt_encoder(
                points=None, boxes=None, masks=None,
                text_embeds=pred_embeddings_,
            )   # [#seg, 1, dim], [#seg, dim, 64, 64]
            sparse_embeddings = sparse_embeddings.to(pred_embeddings_.dtype)
            low_res_masks, _ = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[b:b+1],
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )   # [#seg, 1, h, w]
            pred_mask = self.model.visual_model.postprocess_masks(
                masks=low_res_masks,
                input_size=sam_resized_sizes[b],
                original_size=(height[b], width[b]),
            )   # [#seg, 1, h, w]
            pred_masks.append(pred_mask.squeeze(1))

        # return output_ids, pred_masks, (hidden_states, pred_embeddings, orig_h)
        return output_ids, pred_masks


    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
    ):
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            seg_token_mask = torch.cat(
                [
                    torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(),
                    seg_token_mask,
                ],
                dim=1,
            )

            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]

            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            image_embeddings = self.get_visual_embs(images)

            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )

                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                pred_masks.append(pred_mask[:, 0])

        return output_ids, pred_masks
