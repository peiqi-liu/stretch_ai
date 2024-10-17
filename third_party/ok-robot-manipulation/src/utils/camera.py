# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from dataclasses import dataclass
from typing import Type

import numpy as np
from PIL import Image


@dataclass
class CameraParameters:
    fx: float
    fy: float
    cx: float
    cy: float
    head_tilt: float
    image: Type[Image.Image]
    colors: np.ndarray
    depths: np.ndarray
