# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import List, Optional, Union

import cv2
import numpy as np
import zmq
from utils.types import Number


class ZmqSocket:
    def __init__(self, cfgs):
        # init socket with port number
        zmq_context = zmq.Context()
        self.socket = zmq_context.socket(zmq.REP)
        self.socket.bind("tcp://*:" + str(cfgs.port))

    def send_array(
        self, data: np.ndarray, flags: int = 0, copy: bool = True, track: bool = False
    ) -> Optional[int]:
        """send a numpy array with metadata"""
        md = dict(
            dtype=str(data.dtype),
            shape=data.shape,
        )
        self.socket.send_json(md, flags | zmq.SNDMORE)

        return self.socket.send(np.ascontiguousarray(data), flags, copy=copy, track=track)

    def recv_array(self, flags: int = 0, copy: bool = True, track: bool = False) -> np.ndarray:
        """Receive a NumPy array."""
        md = self.socket.recv_json(flags=flags)
        msg = self.socket.recv(flags=flags, copy=copy, track=track)
        data = np.frombuffer(msg, dtype=md["dtype"])

        return data.reshape(md["shape"])

    def send_data(
        self, data: Union[str, Union[List[Number], List[Union[List[Number], str]]]]
    ) -> Optional[bool]:
        """Send msg - string or list of Numbers or list of list Numbers or strings"""

        # After sending anytype of data other than str it waits for the string confirmation from robot
        if isinstance(data, str):
            self.socket.send_string(data)
        elif isinstance(data, list) and all((not isinstance(num, list)) for num in data):
            data = np.array(data)
            self.send_array(data)
            print(self.recv_string())
        else:
            for d in data:
                if isinstance(d, str):
                    self.socket.send_string(d)
                else:
                    print(d)
                    data = np.array(d)
                    self.send_array(data)
                    print(self.recv_string())

    def recv_string(self) -> str:
        return self.socket.recv_string()

    def send_rgb_img(self, img):
        img = img.astype(np.uint8)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, img_encoded = cv2.imencode(".jpg", img, encode_param)
        self.socket.send(img_encoded.tobytes())

    def recv_rgb_img(self):
        img = self.socket.recv()
        img = np.frombuffer(img, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        return img

    def send_depth_img(self, depth_img):
        depth_img = (depth_img * 1000).astype(np.uint16)
        encode_param = [
            int(cv2.IMWRITE_PNG_COMPRESSION),
            3,
        ]  # Compression level from 0 (no compression) to 9 (max compression)
        _, depth_img_encoded = cv2.imencode(".png", depth_img, encode_param)
        self.socket.send(depth_img_encoded.tobytes())

    def recv_depth_img(self):
        depth_img = self.socket.recv()
        depth_img = np.frombuffer(depth_img, dtype=np.uint8)
        depth_img = cv2.imdecode(depth_img, cv2.IMREAD_UNCHANGED)
        depth_img = depth_img / 1000.0
        return depth_img
