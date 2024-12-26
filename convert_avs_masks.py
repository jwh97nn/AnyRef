import os

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import jaccard_score
from tqdm import tqdm

from model.segment_anything import SamPredictor, build_sam_vit_h
from utils.avsbench import AVSObject

sam = build_sam_vit_h("SAM/sam_vit_h_4b8939.pth")
sam = sam.eval()
sam = sam.cuda()

predictor = SamPredictor(sam)

sam = build_sam_vit_h("SAM/sam_vit_h_4b8939.pth")
sam = sam.eval()
sam = sam.cuda()

for split in ["train", "val", "test"]:
    avs = AVSObject(split=split)
    for i in tqdm(range(len(avs))):
        data = avs.__getitem__(i)
        img = data["file_name"]
        mask = data["gt_masks"][0]

        image = cv2.imread(img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        h, w = image.shape[:2]
        m11 = mask[None, None, ...].float()
        if m11.sum() == 0:
            mask_save = Image.fromarray(np.zeros((h, w), dtype=np.uint8)).convert("P")
            save_path = data["file_name"].replace("visual_frames", "gt_masks")
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            mask_save.save(save_path, format="PNG")
            continue
        m11 = torch.nn.functional.interpolate(
            m11, (h, w), mode="bilinear", align_corners=False
        )[0, 0]
        m11 = (m11 > 0).int()

        m11_np = m11.numpy()
        x1 = np.min(np.where(m11_np > 0)[1])
        x2 = np.max(np.where(m11_np > 0)[1])
        y1 = np.min(np.where(m11_np > 0)[0])
        y2 = np.max(np.where(m11_np > 0)[0])

        input_box = np.array([x1, y1, x2, y2])
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=True,
        )
        iou = jaccard_score(m11_np.flatten(), masks[scores.argmax()].flatten())
        if iou < 0.75:
            mask_save = m11_np.astype(np.uint8) * 255
        else:
            mask_save = masks[scores.argmax()].astype(np.uint8) * 255
        mask_save = Image.fromarray(mask_save).convert("P")
        save_path = data["file_name"].replace("visual_frames", "gt_masks")
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        mask_save.save(save_path, format="PNG")
