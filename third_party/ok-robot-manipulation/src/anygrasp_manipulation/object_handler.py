# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""
 * Codes cited from AnyGrasp: Robust and Efficient Grasp Perception in Spatial and Temporal Domains
 * Author: Fang, Hao-Shu and Wang, Chenxi and Fang, Hongjie and Gou, Minghao and Liu, Jirong and Yan, Hengxu and Liu, Wenhai and Xie, Yichen and Lu, Cewu
 * GitHub: https://github.com/graspnet/anygrasp_sdk
 * All rights reserved by Fang, Hao-Shu.
 *
 * Modifications were made for integration purposes.
"""

import copy
import math
import os
import time

import numpy as np
import open3d as o3d
from graspnetAPI import GraspGroup
from gsnet import AnyGrasp
from image_processors import LangSAMProcessor
from PIL import Image
from utils.camera import CameraParameters
from utils.types import Bbox
from utils.utils import draw_rectangle, get_3d_points, sample_points, visualize_cloud_geometries
from utils.zmq_socket import ZmqSocket

# from image_processors import OWLSAMProcessor


class ObjectHandler:
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.grasping_model = AnyGrasp(self.cfgs)
        self.grasping_model.load_net()

        if self.cfgs.open_communication:
            self.socket = ZmqSocket(self.cfgs)

        # self.lang_sam = OWLSAMProcessor()
        self.lang_sam = LangSAMProcessor()
        self.rerun_frame = 1

    def receive_input(self, tries):
        if self.cfgs.open_communication:
            print("\n\nWaiting for data from Robot")
            # Reading color array
            colors = self.socket.recv_rgb_img()
            self.socket.send_data("RGB received")

            # Depth data
            depths = self.socket.recv_depth_img()
            # print(np.max(depths), np.min(depths))
            self.socket.send_data("depth received")

            # Camera Intrinsics
            fx, fy, cx, cy, head_tilt = self.socket.recv_array()
            self.socket.send_data("intrinsics received")

            # Object query
            self.query = self.socket.recv_string()
            self.socket.send_data("text query received")
            print(f"Text - {self.query}")

            # action -> ["pick", "place"]
            self.action = self.socket.recv_string()
            self.socket.send_data("Mode received")
            print(f"Manipualtion Mode - {self.action}")
            print(self.socket.recv_string())

            image = Image.fromarray(colors)
        else:
            head_tilt = -45
            data_dir = "./example_data/"
            colors = np.array(Image.open(os.path.join(data_dir, "test_rgb.png")))
            image = Image.open(os.path.join(data_dir, "test_rgb.png"))
            # depths = np.array(Image.open(os.path.join(data_dir, "test_depth.png")))
            depths = np.load(os.path.join(data_dir, "test_depth.npy")).astype(np.float64)
            # fx, fy, cx, cy, scale = 215.72599792, 215.7259979, 122.00137329, 155.78569031, 0.001
            fx, fy, cx, cy = 606.4227, 606.4227, 254.01973, 320.10287
            if tries == 1:
                self.action = str(input("Enter action [pick/place]: "))
                self.query = str(input("Enter a Object name in the scene: "))
            # depths = depths * scale

        # Camera Parameters
        colors = colors / 255.0
        head_tilt = head_tilt / 100
        self.cam = CameraParameters(fx, fy, cx, cy, head_tilt, image, colors, depths)

        # rr.set_time_sequence('frame', self.rerun_frame)
        # self.rerun_frame += 1

        # if self.action == 'pick':
        #     rr.init("Picking", spawn = True)
        #     # rr.connect('100.108.67.79:9876')
        #     rr.log('all_anygrasp_estimated_poses', rr.Clear(recursive = True), static = True)
        #     rr.log('Image_received_by_robot', rr.Clear(recursive = True), static = True)
        #     rr.log('robot_monologue', rr.Clear(recursive = True), static = True)
        #     rr.log('selected_pose', rr.Clear(recursive = True), static = True)
        #     rr.log('object_detecion_results', rr.Clear(recursive = True), static = True)
        # else:
        #     rr.init("Placing", spawn = True)
        #     # rr.connect('100.108.67.79:9876')
        #     rr.log('Image_received_by_robot', rr.Clear(recursive = True), static = True)
        #     rr.log('robot_monologue', rr.Clear(recursive = True), static = True)
        #     rr.log('proposed_placing_location', rr.Clear(recursive = True), static = True)
        #     rr.log('object_detecion_results', rr.Clear(recursive = True), static = True)

    def manipulate(self):
        """
        Wrapper for grasping and placing

        11 is the maximum number of retries in case of object or grasp detection failure
        Try - 1 -> captures image and centers the robot
        Try - 2 -> captures image and tries to perform action
        If failed:
            Try - 3,4 -> tries in a different camera orientation
        Even then if it fails:
            Try - 5,6,7 -> moves base to left tries again three different camera orientations
        Even then if it fails:
            Try - 8,9,10 -> moves base to the right and again tries three different camera orientations
        Finally if it fails to detect any pose in above all attempts:
            Try 11 -> If object is detected but anygrasp couldn't find a pose as a last resort
                      the cropped object image is sent to the model.
        In any of the above attempts it is able to succeed it won't perform any further tries
        """

        tries = 1
        retry = True
        while retry and tries <= 11:
            self.receive_input(tries)

            # Directory for saving visualisations
            self.save_dir = self.cfgs.environment + "/" + self.query + "/anygrasp/"
            debug_text = (
                "### Robot's monolouge: \n ## The text query I received is " + self.query + ".\n"
            )
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            if self.cfgs.open_communication:
                camera_image_file_name = self.save_dir + "clean_" + str(tries) + ".jpg"
                self.cam.image.save(camera_image_file_name)
                # rr.log('Image_received_by_robot', rr.Image(cv2.imread(camera_image_file_name)[:, :, [2, 1, 0]]), static = True)
                print(f"Saving the camera image at {camera_image_file_name}")
                np.save(self.save_dir + "depths_" + str(tries) + ".npy", self.cam.depths)

            box_filename = (
                f"{self.cfgs.environment}/{self.query}/anygrasp/object_detection_{tries}.jpg"
            )
            mask_filename = (
                f"{self.cfgs.environment}/{self.query}/anygrasp/semantic_segmentation_{tries}.jpg"
            )
            # Object Segmentation Mask

            colors = np.array(self.cam.image)
            colors[self.cam.depths > self.cfgs.max_depth + 0.3] = 1e-4
            colors = Image.fromarray(colors, "RGB")
            colors.save(camera_image_file_name)

            seg_mask, bbox = self.lang_sam.detect_obj(
                self.cam.image,
                self.query,
                visualize_box=True,
                visualize_mask=True,
                box_filename=box_filename,
                mask_filename=mask_filename,
            )

            if bbox is None:
                if self.cfgs.open_communication:
                    print("Didn't detect the object, Trying Again")
                    tries = tries + 1
                    print(f"Try no: {tries}")
                    data_msg = "No Objects detected, Have to try again"
                    self.socket.send_data([[0], [0], [0, 0, 2], data_msg])
                    if tries == 11:
                        return
                    continue
                    # self.socket.send_data("No Objects detected, Have to try again")
                else:
                    print(
                        "Didn't find the Object. Try with another object or tune grasper height and width parameters in demo.py"
                    )
                    retry = False
                    continue

            bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = bbox

            # Center the robot
            if tries == 1 and self.cfgs.open_communication:
                # rr.log("robot_monologue", rr.TextDocument(debug_text + '### Now I move myself such that ' + self.query + ' appears in the center of the head camera.\n', media_type = rr.MediaType.MARKDOWN), static = True)
                self.center_robot(bbox)
                tries += 1
                time.sleep(2.5)
                continue

            points = get_3d_points(self.cam)

            if self.action == "place":
                # rr.log("robot_monologue", rr.TextDocument(debug_text + '### Now I will try to place the object in my gripper on ' + self.query + '.\n ### Firstly, LangSAM will detect ' + self.query + '. \n ### Then, the middle of segmentation mask will be selected as the placing location.\n', media_type = rr.MediaType.MARKDOWN), static = True)

                retry = not self.place(points, seg_mask)
            else:
                # rr.log("robot_monologue", rr.TextDocument(debug_text + '### Now I will try to pick up ' + self.query + '.\n ### Firstly, AnyGrasp will generate collision free grasps of all objects in the scene. \n ### After that, LangSAM will detect ' + self.query + '.\n ### Finally, I will filter AnyGrasp poses based on segmentation mask and select the best gripper pose.', media_type = rr.MediaType.MARKDOWN), static = True)

                retry = not self.pickup(points, seg_mask, bbox, (tries == 11))

            if retry:
                if self.cfgs.open_communication:
                    print("Trying Again")
                    tries = tries + 1
                    print(f"Try no: {tries}")
                    data_msg = "No poses, Have to try again"
                    self.socket.send_data([[0], [0], [0, 0, 2], data_msg])
                    # self.socket.send_data("No poses, Have to try again")
                else:
                    print(
                        "Try with another object or tune grasper height and width parameters in demo.py"
                    )
                    retry = False

    def center_robot(self, bbox: Bbox):
        """
        Center the robots base and camera to face the center of the Object Bounding box
        """

        bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = bbox

        bbox_center = [
            int((bbox_x_min + bbox_x_max) / 2),
            int((bbox_y_min + bbox_y_max) / 2),
        ]
        depth_obj = self.cam.depths[bbox_center[1], bbox_center[0]]
        print(
            f"{self.query} height and depth: {((bbox_y_max - bbox_y_min) * depth_obj)/self.cam.fy}, {depth_obj}"
        )

        # base movement
        dis = (bbox_center[0] - self.cam.cx) / self.cam.fx * depth_obj
        print(f"Base displacement {dis}")

        # camera tilt
        tilt = math.atan((bbox_center[1] - self.cam.cy) / self.cam.fy)
        print(f"Camera Tilt {tilt}")

        if self.cfgs.open_communication:
            data_msg = "Now you received the base and head trans, good luck."
            self.socket.send_data([[-dis], [-tilt], [0, 0, 1], data_msg])
            # self.socket.send_data("Now you received the base and head trans, good luck.")

    def place(self, points: np.ndarray, seg_mask: np.ndarray) -> bool:
        points_x, points_y, points_z = points[:, :, 0], points[:, :, 1], points[:, :, 2]
        flat_x, flat_y, flat_z = (
            points_x.reshape(-1),
            -points_y.reshape(-1),
            -points_z.reshape(-1),
        )

        # Removing all points whose depth is zero(undetermined)
        zero_depth_seg_mask = (
            (flat_x != 0)
            * (flat_y != 0)
            * (flat_z != 0)
            * (~np.isnan(flat_z))
            * seg_mask.reshape(-1)
        )
        flat_x = flat_x[zero_depth_seg_mask]
        flat_y = flat_y[zero_depth_seg_mask]
        flat_z = flat_z[zero_depth_seg_mask]

        colors = self.cam.colors.reshape(-1, 3)[zero_depth_seg_mask]

        # 3d point cloud in camera orientation
        points1 = np.stack([flat_x, flat_y, flat_z], axis=-1)

        # Rotation matrix for camera tilt
        cam_to_3d_rot = np.array(
            [
                [1, 0, 0],
                [0, math.cos(self.cam.head_tilt), math.sin(self.cam.head_tilt)],
                [0, -math.sin(self.cam.head_tilt), math.cos(self.cam.head_tilt)],
            ]
        )

        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(points1)
        pcd1.colors = o3d.utility.Vector3dVector(colors)

        # 3d point cloud with upright camera
        transformed_points = np.dot(points1, cam_to_3d_rot)

        # Removing floor points from point cloud
        floor_mask = transformed_points[:, 1] > -1.25
        transformed_points = transformed_points[floor_mask]
        transformed_x = transformed_points[:, 0]
        transformed_y = transformed_points[:, 1]
        transformed_z = transformed_points[:, 2]
        colors = colors[floor_mask]

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(transformed_points)
        pcd2.colors = o3d.utility.Vector3dVector(colors)

        # Projected Median in the xz plane [parallel to floor]
        xz = np.stack([transformed_x * 100, transformed_z * 100], axis=-1).astype(int)
        unique_xz = np.unique(xz, axis=0)
        unique_xz_x, unique_xz_z = unique_xz[:, 0], unique_xz[:, 1]
        px, pz = np.median(unique_xz_x) / 100.0, np.median(unique_xz_z) / 100.0

        x_margin, z_margin = 0.1, 0
        x_mask = (transformed_x < (px + x_margin)) & (transformed_x > (px - x_margin))
        y_mask = (transformed_y < 0) & (transformed_y > -1.1)
        z_mask = (transformed_z < 0) & (transformed_z > (pz - z_margin))
        mask = x_mask & y_mask & z_mask
        py = np.max(transformed_y[mask])
        point = np.array([px, py, pz])  # Final placing point in upright camera coordinate system

        geometries = []
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.1, height=0.04)
        cylinder_rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        cylinder.rotate(cylinder_rot)
        cylinder.translate(cam_to_3d_rot @ point)
        cylinder.rotate(cam_to_3d_rot)
        cylinder.paint_uniform_color([0, 1, 0])
        geometries.append(cylinder)

        if self.cfgs.debug:
            visualize_cloud_geometries(
                pcd1,
                geometries,
                save_file=self.save_dir + "/placing.jpg",
                visualize=not self.cfgs.headless,
                rerun_name="proposed_placing_location",
            )

        point[1] += 0.1
        transformed_point = cam_to_3d_rot @ point
        print(f"Placing point of Object relative to camera: {transformed_point}")

        if self.cfgs.open_communication:
            data_msg = "Now you received the gripper pose, good luck."
            self.socket.send_data(
                [
                    np.array(transformed_point, dtype=np.float64),
                    [0],
                    [0, 0, 0],
                    data_msg,
                ]
            )

        return True

    def pickup(
        self,
        points: np.ndarray,
        seg_mask: np.ndarray,
        bbox: Bbox,
        crop_flag: bool = False,
    ):
        points_x, points_y, points_z = points[:, :, 0], points[:, :, 1], points[:, :, 2]

        # Filtering points based on the distance from camera
        mask = (
            (points_z > self.cfgs.min_depth)
            & (points_z < self.cfgs.max_depth)
            & ~np.isnan(points_z)
        )
        points = np.stack([points_x, -points_y, points_z], axis=-1)
        points = points[mask].astype(np.float32)
        colors_m = self.cam.colors[mask].astype(np.float32)

        if self.cfgs.sampling_rate < 1:
            points, indices = sample_points(points, self.cfgs.sampling_rate)
            colors_m = colors_m[indices]

        # Grasp Prediction
        # gg is a list of grasps of type graspgroup in graspnetAPI
        xmin = points[:, 0].min()
        xmax = points[:, 0].max()
        ymin = points[:, 1].min()
        ymax = points[:, 1].max()
        zmin = points[:, 2].min()
        zmax = points[:, 2].max()
        lims = [xmin, xmax, ymin, ymax, zmin, zmax]
        gg, cloud = self.grasping_model.get_grasp(points, colors_m, lims)

        if len(gg) == 0:
            print("No Grasp detected after collision detection!")
            return False

        gg = gg.nms().sort_by_score()
        filter_gg = GraspGroup()

        # Filtering the grasps by penalising the vertical grasps as they are not robust to calibration errors.
        bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = bbox
        W, H = self.cam.image.size
        ref_vec = np.array([0, math.cos(self.cam.head_tilt), -math.sin(self.cam.head_tilt)])

        # R_tilt = np.array([
        #     [1, 0, 0],
        #     [0, math.cos(self.cam.head_tilt), -math.sin(self.cam.head_tilt)],
        #     [0, math.sin(self.cam.head_tilt), math.cos(self.cam.head_tilt)]
        # ])

        min_score, max_score = 1, -10
        image = copy.deepcopy(self.cam.image)
        img_drw = draw_rectangle(image, bbox)
        for g in gg:
            grasp_center = g.translation
            # z_value = (grasp_center @ R_tilt.T)[0]
            # print('z_value', z_value)
            ix, iy = (
                int(((grasp_center[0] * self.cam.fx) / grasp_center[2]) + self.cam.cx),
                int(((-grasp_center[1] * self.cam.fy) / grasp_center[2]) + self.cam.cy),
            )
            if ix < 0:
                ix = 0
            if iy < 0:
                iy = 0
            if ix >= W:
                ix = W - 1
            if iy >= H:
                iy = H - 1
            rotation_matrix = g.rotation_matrix
            cur_vec = rotation_matrix[:, 0]
            angle = math.acos(np.dot(ref_vec, cur_vec) / (np.linalg.norm(cur_vec)))
            if not crop_flag:
                score = g.score - 0.5 * (angle - 0.5) ** 4
                # score = - 100 * (angle - 0.3) ** 4
                # print(0.1 * (angle) ** 4, g.score)
            else:
                score = g.score

            if not crop_flag:
                if seg_mask[iy, ix]:
                    img_drw.ellipse([(ix - 2, iy - 2), (ix + 2, iy + 2)], fill="green")
                    if g.score >= 0.095:
                        g.score = score
                    min_score = min(min_score, g.score)
                    max_score = max(max_score, g.score)
                    filter_gg.add(g)
                else:
                    img_drw.ellipse([(ix - 2, iy - 2), (ix + 2, iy + 2)], fill="red")
            else:
                g.score = score
                filter_gg.add(g)

        if len(filter_gg) == 0:
            print(
                "No grasp poses detected for this object try to move the object a little and try again"
            )
            return False

        projections_file_name = (
            self.cfgs.environment + "/" + self.query + "/anygrasp/grasp_projections.jpg"
        )
        image.save(projections_file_name)
        print(f"Saved projections of grasps at {projections_file_name}")
        filter_gg = filter_gg.nms().sort_by_score()

        if self.cfgs.debug:
            trans_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            cloud.transform(trans_mat)
            grippers = gg.to_open3d_geometry_list()
            filter_grippers = filter_gg.to_open3d_geometry_list()
            for gripper in grippers:
                gripper.transform(trans_mat)
            for gripper in filter_grippers:
                gripper.transform(trans_mat)

            visualize_cloud_geometries(
                cloud,
                grippers,
                visualize=not self.cfgs.headless,
                save_file=f"{self.cfgs.environment}/{self.query}/anygrasp/poses.jpg",
                rerun_name="all_anygrasp_estimated_poses",
            )
            visualize_cloud_geometries(
                cloud,
                [filter_grippers[0].paint_uniform_color([1.0, 0.0, 0.0])],
                visualize=not self.cfgs.headless,
                save_file=f"{self.cfgs.environment}/{self.query}/anygrasp/best_pose.jpg",
                rerun_name="selected_pose",
            )

        if self.cfgs.open_communication:
            data_msg = "Now you received the gripper pose, good luck."
            self.socket.send_data(
                [
                    filter_gg[0].translation,
                    filter_gg[0].rotation_matrix,
                    [filter_gg[0].depth, filter_gg[0].width, 0],
                    data_msg,
                ]
            )
        return True
