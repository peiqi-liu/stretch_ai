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
import logging

import numpy as np
import trimesh.transformations as tra
from geometry_msgs.msg import Point, Pose, Quaternion, Transform

log = logging.getLogger(__name__)


def theta_to_quaternion_msg(theta):
    T = tra.euler_matrix(0, 0, theta)
    w, x, y, z = tra.quaternion_from_matrix(T)
    return Quaternion(x, y, z, w)


def quaternion_msg_to_theta(msg):
    T = tra.quaternion_matrix([msg.w, msg.x, msg.y, msg.z])
    a, b, theta = tra.euler_from_matrix(T)
    print("quat msg to theta --", a, b, theta)
    return theta


def matrix_from_pose_msg(msg):
    T = tra.quaternion_matrix(
        [msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z]
    )
    T[:3, 3] = np.array([msg.position.x, msg.position.y, msg.position.z])
    return T


def matrix_to_pose_msg(matrix):
    pose = Pose()
    w, x, y, z = tra.quaternion_from_matrix(matrix)

    # Direct initialization not working for Quaternion and Point -> Quaternion(x, y, z, w) and Point(x, y, z)
    pose.orientation = Quaternion(x=x * 1.0, y=y * 1.0, z=z * 1.0, w=w * 1.0)
    xyz = matrix[:3, 3].tolist()
    pose.position = Point(x=xyz[0], y=xyz[1], z=xyz[2])
    return pose


def ros_pose_to_transform(pose_msg):
    t = Transform()
    t.translation.x = pose_msg.position.x
    t.translation.y = pose_msg.position.y
    t.translation.z = pose_msg.position.z
    t.rotation.x = pose_msg.orientation.x
    t.rotation.y = pose_msg.orientation.y
    t.rotation.z = pose_msg.orientation.z
    t.rotation.w = pose_msg.orientation.w
    return t


def to_normalized_quaternion_msg(w, x, y, z):
    quat = np.array([w, x, y, z])
    quat = quat / np.linalg.norm(quat)
    return Quaternion(x, y, z, w)
