# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import os

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():

    realsense_config = {
        "align_depth.enable": "True",
        "camera_name": "camera",
        "camera_namespace": "",
        # "decimation_filter.enable": "True",
        # "spatial_filter.enable": "True",
        # "temporal_filter.enable": "True",
        # "disparity_filter.enable": "False",
        "device_type": "d435i",
        "rgb_camera.color_profile": "640x480x30",
        "depth_module.depth_profile": "640x480x30",
        "depth_module.infra_profile": "640x480x30",
        "enable_gyro": "true",
        "enable_accel": "true",
        "gyro_fps": "200",
        "accel_fps": "100",
    }
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("stretch_ros2_bridge"), "launch/rs_launch.py")
        ),
        launch_arguments=realsense_config.items(),
    )

    ld = LaunchDescription(
        [
            realsense_launch,
        ]
    )
    return ld
