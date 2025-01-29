# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import argparse
import os
import sys

from anygrasp_manipulation import ObjectHandler

FAIL_COLOR = "\033[91m"
NO_LICENSE_MSG = f"""
{FAIL_COLOR}Couldn't find the license folder in the /src directory. 
Check the readme at https://github.com/graspnet/anygrasp_sdk?tab=readme-ov-file#license-registration 
to get a license for yourself.

If you already have a license, group the license related .json, .lic, .public_key, 
.signature files into a "license" folder and place it inside the /src directory
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint_path",
    default="./checkpoints/checkpoint_detection.tar",
    help="Model checkpoint path",
)
parser.add_argument(
    "--max_gripper_width",
    type=float,
    default=0.07,
    help="Maximum gripper width (<=0.1m)",
)
parser.add_argument("--gripper_height", type=float, default=0.05, help="Gripper height")
parser.add_argument("--port", type=int, default=5556, help="port")
parser.add_argument("--top_down_grasp", action="store_true", help="Output top-down grasps")
parser.add_argument("--debug", action="store_true", help="Enable visualization")
parser.add_argument("--headless", action="store_true", help="Enable headless mode")
parser.add_argument(
    "--open_communication",
    action="store_true",
    help="Use image transferred from the robot",
)
parser.add_argument("--max_depth", type=float, default=1.5, help="Maximum depth of point cloud")
parser.add_argument("--min_depth", type=float, default=0.1, help="Maximum depth of point cloud")
parser.add_argument(
    "--sampling_rate", type=float, default=0.8, help="Sampling rate of points [<= 1]"
)
parser.add_argument("--environment", default="./dynamem_log", help="Environment name")
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.2, cfgs.max_gripper_width))


def check_license_folder():
    license_path = "./license"
    if (not os.path.exists(license_path)) or (len(os.listdir(license_path)) < 4):
        print(NO_LICENSE_MSG)
        sys.exit(1)


def demo():
    # Checking the proper license folder placement.
    check_license_folder()
    # rr.init('Demo', spawn = False)
    # rr.connect('100.108.67.79:9876')

    object_handler = ObjectHandler(cfgs)
    while True:
        object_handler.manipulate()


if __name__ == "__main__":
    demo()
