# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import List, Tuple, Type

import cv2
import numpy as np
import rerun as rr
from image_processors.image_processor import ImageProcessor
from lang_sam import LangSAM
from PIL import Image


class LangSAMProcessor(ImageProcessor):
    def __init__(self):
        super().__init__()

        self.model = LangSAM()

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
        masks, boxes, phrases, logits = self.model.predict(image, text)
        if len(masks) == 0:
            return masks, None

        seg_mask = np.array(masks[0])
        bbox = np.array(boxes[0], dtype=int)

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
