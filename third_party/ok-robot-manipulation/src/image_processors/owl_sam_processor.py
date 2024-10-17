# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import os
from typing import List, Tuple, Type

import cv2
import numpy as np
import rerun as rr
import torch
import wget
from image_processors.image_processor import ImageProcessor
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, Owlv2ForObjectDetection


class OWLSAMProcessor(ImageProcessor):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device

        self.processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16-ensemble"
        ).to(self.device)

        sam_checkpoint = f"./sam2_hiera_small.pt"
        sam_config = "sam2_hiera_s.yaml"
        if not os.path.exists(sam_checkpoint):
            wget.download(
                "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
                out=sam_checkpoint,
            )
        sam2_model = build_sam2(
            sam_config, sam_checkpoint, device=self.device, apply_postprocessing=False
        )
        self.mask_predictor = SAM2ImagePredictor(sam2_model)

    def detect_obj(
        self,
        image: Type[Image.Image],
        text: str = None,
        bbox: List[int] = None,
        visualize_box: bool = False,
        box_filename: str = None,
        visualize_mask: bool = False,
        mask_filename: str = None,
    ) -> Tuple[np.ndarray, List[int]]:
        print("OWLSAM detection !!!")
        inputs = self.processor(text=[["a photo of a " + text]], images=image, return_tensors="pt")
        for input in inputs:
            inputs[input] = inputs[input].to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.Tensor([image.size[::-1]]).to("cuda")
        results = self.processor.image_processor.post_process_object_detection(
            outputs, threshold=0.05, target_sizes=target_sizes
        )[0]

        if len(results["boxes"]) == 0:
            return None, None

        bounding_box = results["boxes"][torch.argmax(results["scores"])]

        bounding_boxes = bounding_box.unsqueeze(0)

        self.mask_predictor.set_image(image)
        masks, _, _ = self.mask_predictor.predict(
            point_coords=None, point_labels=None, box=bounding_boxes, multimask_output=False
        )
        if len(masks) == 0:
            return None, None
        mask = torch.Tensor(masks).bool()[0]

        seg_mask = mask.detach().cpu().numpy()
        bbox = np.array(bounding_box.detach().cpu(), dtype=int)

        # if visualize_box:
        #     self.draw_bounding_box(image, bbox, box_filename)
        #     if box_filename is not None:
        #         rr.log('object_detection_results', rr.Image(cv2.imread(box_filename)[:, :, [2, 1, 0]]), static = False)

        if visualize_mask:
            self.draw_bounding_box(image, bbox, box_filename)
            self.draw_mask_on_image(image, seg_mask, mask_filename)
            if mask_filename is not None:
                rr.log(
                    "object_detection_results",
                    rr.Image(cv2.imread(mask_filename)[:, :, [2, 1, 0]]),
                    static=True,
                )

        return seg_mask, bbox
