# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Load data and visualize it
"""


class Camera:
    """Camera object storing information about a single camera.
    Includes projection and pose information."""

    def __init__(
        self,
        pos=None,
        orn=None,
        height=None,
        width=None,
        fx=None,
        fy=None,
        px=None,
        py=None,
        near_val=None,
        far_val=None,
        pose_matrix=None,
        proj_matrix=None,
        view_matrix=None,
        fov=None,
        *args,
        **kwargs
    ):
        self.pos = pos
        self.orn = orn
        self.height = height
        self.width = width
        self.px = px
        self.py = py
        self.fov = fov
        self.near_val = near_val
        self.far_val = far_val
        self.fx = fx
        self.fy = fy
        self.pose_matrix = pose_matrix
        self.pos = pos
        self.orn = orn
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self):
        """create a dictionary so that we can extract the necessary information for
        creating point clouds later on if we so desire"""
        info = {}
        info["pos"] = self.pos
        info["orn"] = self.orn
        info["height"] = self.height
        info["width"] = self.width
        info["near_val"] = self.near_val
        info["far_val"] = self.far_val
        # info['proj_matrix'] = self.proj_matrix
        # info['view_matrix'] = self.view_matrix
        # info['max_depth'] = self.max_depth
        info["pose_matrix"] = self.pose_matrix
        info["px"] = self.px
        info["py"] = self.py
        info["fx"] = self.fx
        info["fy"] = self.fy
        info["fov"] = self.fov
        return info

    def get_pose(self):
        return self.pose_matrix.copy()
