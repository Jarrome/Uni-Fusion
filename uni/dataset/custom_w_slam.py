import os, sys
import torch
import cv2
import numpy as np
import open3d as o3d
from PIL import Image
import tensorflow.compat.v1 as tf
from threading import Thread, Condition, Lock
from pyquaternion import Quaternion


from uni.dataset import *
from uni.dataset.scannet import ScanNetRGBDDataset
from uni.dataset.replica import ReplicaRGBDDataset
from uni.dataset.bpnet_scannet import ScanNetLatentDataset
from uni.dataset.azure import AzureRGBDIDataset
from uni.dataset.tum import TUMRGBDDataset

from uni.utils import motion_util
import orbslam2
from tqdm import tqdm

import glob


from time import sleep
import pdb

def read_orbslam2_file(traj_file):
    with open(traj_file) as f:
        lines = f.readlines()
    poses = []
    frame_ids = []
    for line_id, line in enumerate(lines):
        vs = [float(v) for v in line.strip().split(' ')]
        frame_id = round(vs[0])
        v_t = vs[1:4]
        #v_q = vs[4:] # xyzw
        v_q = Quaternion(vs[-1],*vs[4:-1])
        pose = v_q.transformation_matrix
        pose[:3,3] = np.array(v_t)
        poses.append(pose)
        frame_ids.append(frame_id)
    return frame_ids, poses


class SLAM():
    def __init__(self, 
            vocab_file= './external/Uni-Fusion-use-ORB-SLAM2/Vocabulary/ORBvoc.bin',
            setting_file = './external/Uni-Fusion-use-ORB-SLAM2/Examples/RGB-D/azure_office.yaml',
            mode = 'rgbd',
            init_pose = np.eye(4,dtype=np.float32),
            use_viewer = False):
        self.slam = orbslam2.SLAM(vocab_file, setting_file, mode, init_pose, use_viewer)

        self.poses = []

    def feed_stack_and_start_thread(self, rgbs, depths):
        self.rgbs = rgbs
        self.depth = depths
        self.maintenance_thread = Thread(target=self.maintenance)
        self.maintenance_thread.daemon = True # make sure main thread can exit
        self.maintenance_thread.start()


    def maintenance(self):
        for frame_id in range(len(self.rgbs)):
            rgb = self.rgbs[frame_id]
            depth = self.depth[frame_id].astype(np.float32)
            pose = self.slam.track(rgb, depth) 
            self.poses.append(np.linalg.inv(pose))

        self.slam.shutdown()




 
class CustomAzurewSLAM(AzureRGBDIDataset):
    def __init__(self, path, start_frame: int = 0, end_frame: int = -1, first_tq: list = None, load_gt: bool = False, train=True, mesh_gt = None, style_idx=1, has_style=True, has_ir=False, has_saliency=True, has_latent=False, f_im=None, slam=False):
        super().__init__(path, start_frame, end_frame, first_tq, load_gt, train, mesh_gt)
        if slam and not load_gt:
            self.slam = SLAM()
            self.slam.feed_stack_and_start_thread(self.rgb, self.depth)

        self.has_ir = has_ir
        self.has_saliency = has_saliency
        self.has_style = has_style
        self.has_latent = has_latent
        # saliency
        if has_saliency:
            from transparent_background import Remover 
            self.saliency_detector = Remover()

        # style
        if has_style:
            from external.style_transfer.experiments import style_api
            self.style_painting = style_api.get_api(style_idx)

        # latent
        self.latent_func = f_im

        # np_str
        if has_latent:
            self.np_image_strings = []
            for rgb_id in tqdm(self.rgb_ids):
                with tf.gfile.GFile(rgb_id, 'rb') as f:
                    np_image_string = np.array([f.read()])
                    self.np_image_strings.append(np_image_string)

        self.first_iso = motion_util.Isometry(q=Quaternion(array=[0.0, -1.0, 0.0, 0.0]))
        self.change_iso = None

    def orbslam2pose_to_LIMpose(self, pose):
        try:
            cur_iso = motion_util.Isometry(q=Quaternion(matrix=pose[:3,:3], atol=1e-5, rtol=1e-5), t=pose[:3,3])
        except Exception as e:
            print(pose, e)
            return None
        if self.change_iso is None:
            self.change_iso = self.first_iso.dot(cur_iso.inv())
            self.T_gt2uni = self.change_iso.matrix
        return self.change_iso.dot(cur_iso)
        

    def __getitem__(self, idx): 
        frame_data = FrameData()
        if hasattr(self,'slam'):
            while idx >= len(self.slam.poses):
                sleep(.1)

            frame_data.gt_pose = self.orbslam2pose_to_LIMpose(self.slam.poses[idx])
            '''
        while frame_data.gt_pose is None:
            idx += 1
            frame_data.gt_pose = self.orbslam2pose_to_LIMpose(self.slam.poses[idx])
            '''
        else:
            if self.gt_trajectory is not None:
                frame_data.gt_pose = self.gt_trajectory[idx]
            else:
                frame_data.gt_pose = None


        frame_data.calib = FrameIntrinsic(self.cam.fx, self.cam.fy, self.cam.cx, self.cam.cy, self.cam.scale)
        frame_data.depth =  torch.from_numpy(self.depth[idx].astype(np.float32)).cuda(0).float() / self.cam.scale
        frame_data.rgb = torch.from_numpy(self.rgb[idx]).cuda(0).float() / 255.
        frame_data.ir = torch.from_numpy(self.ir[idx].astype(np.float32)).cuda(0).float().unsqueeze(-1) if self.has_ir else None
        
        img = Image.fromarray(self.rgb[idx]).convert('RGB')
        frame_data.saliency = torch.from_numpy(
                self.saliency_detector.process(img,type='map')
                ).cuda(0).float() /255. if self.has_saliency else None



        frame_data.style = torch.from_numpy(
                self.style_painting(img)).cuda(0).float() / 255. if self.has_style else None

        H,W,_ = frame_data.rgb.shape

        frame_data.latent = torch.from_numpy(self.latent_func(self.np_image_strings[idx], H, W)).cuda(0).float() if self.has_latent else None


        frame_data.customs = [frame_data.ir, frame_data.saliency, frame_data.style, frame_data.latent]
        return frame_data

class CustomReplicawSLAM(ReplicaRGBDDataset):
    def __init__(self, path, start_frame: int = 0, end_frame: int = -1, first_tq: list = None, load_gt: bool = False, train=True, mesh_gt = None, style_idx=1, has_style=False, has_ir=False, has_saliency=False, has_latent=False, f_im=None, slam=False):
        load_file_gt = load_gt #False
        super().__init__(path, start_frame, end_frame, first_tq, load_file_gt, train, mesh_gt)
        if slam:
            self.slam = SLAM(setting_file='./external/Uni-Fusion-use-ORB-SLAM2/Examples/RGB-D/replica.yaml')
            if not load_gt:
                self.slam.feed_stack_and_start_thread(self.rgb, self.depth)
            else:
                # for rerun the predicted trajectory
                # load from predicted trajectory 'pred_traj.txt'
                frame_ids, poses = read_orbslam2_file(path+'/../pred_traj.txt')
                # fill in not listed poses
                now_id = 0
                frame_ids_ = []
                poses_ = []
                for idx in range(0,frame_ids[-1]+1):
                    frame_ids_.append(idx)
                    if idx in frame_ids:
                        poses_.append(poses[now_id])
                        now_id += 1
                    else:
                        poses_.append(np.eye(4))
                frame_ids = frame_ids_
                self.slam.poses = poses_


        self.has_ir = has_ir
        self.has_saliency = has_saliency
        self.has_style = has_style
        self.has_latent = has_latent
        # saliency
        if has_saliency:
            from transparent_background import Remover 
            self.saliency_detector = Remover()

        # style
        if has_style:
            from external.style_transfer.experiments import style_api
            self.style_painting = style_api.get_api(style_idx)

        # latent
        self.latent_func = f_im

        # np_str
        if has_latent:
            self.np_image_strings = []
            for rgb_id in tqdm(self.rgb_ids):
                with tf.gfile.GFile(rgb_id, 'rb') as f:
                    np_image_string = np.array([f.read()])
                    self.np_image_strings.append(np_image_string)

        self.first_iso = motion_util.Isometry(q=Quaternion(array=[0.0, -1.0, 0.0, 0.0]))
        self.change_iso = None

    def orbslam2pose_to_LIMpose(self, pose):
        try:
            cur_iso = motion_util.Isometry(q=Quaternion(matrix=pose[:3,:3], atol=1e-5, rtol=1e-5), t=pose[:3,3])
        except Exception as e:
            print(pose, e)
            return None
        if self.change_iso is None:
            self.change_iso = self.first_iso.dot(cur_iso.inv())
            self.T_gt2uni = self.change_iso.matrix
        return self.change_iso.dot(cur_iso)
        

    def __getitem__(self, idx): 
        frame_data = FrameData()
        #frame_data.gt_pose = self.gt_trajectory[idx]
        if hasattr(self,'slam'):
            # wait slam to track
            while idx >= len(self.slam.poses):
                sleep(.1)

            frame_data.gt_pose = self.orbslam2pose_to_LIMpose(self.slam.poses[idx])
        else:
            if self.gt_trajectory is not None:
                frame_data.gt_pose = self.gt_trajectory[idx]
            else:
                frame_data.gt_pose = None



        frame_data.calib = FrameIntrinsic(600., 600., 599.5, 339.5, 6553.5)
        frame_data.depth =  torch.from_numpy(self.depth[idx].astype(np.float32)).cuda(0).float() / 6553.5
        frame_data.rgb = torch.from_numpy(self.rgb[idx]).cuda(0).float() / 255.

        img = Image.fromarray((frame_data.rgb.cpu().numpy()*255).astype(np.ubyte)).convert('RGB')

        frame_data.ir = None
        frame_data.saliency = torch.from_numpy(
                self.saliency_detector.process(img,type='map').astype(np.float32)
                ).cuda(0).float() / 255. if self.has_saliency else None

        frame_data.style = torch.from_numpy(
                self.style_painting(img)).cuda(0).float() / 255. if self.has_style else None


        H,W,_ = frame_data.rgb.shape

        frame_data.latent = torch.from_numpy(self.latent_func(self.np_image_strings[idx], H, W)).cuda(0).float() if self.has_latent else None


        frame_data.customs = [frame_data.ir, frame_data.saliency, frame_data.style, frame_data.latent]
        return frame_data



class CustomScanNetwSLAM(ScanNetRGBDDataset):
    def __init__(self, path, start_frame: int = 0, end_frame: int = -1, first_tq: list = None, load_gt: bool = False, train=True, mesh_gt = None, style_idx=1, has_style=True, has_ir=False, has_saliency=True, has_latent=False, f_im=None, slam=False):
        super().__init__(path, start_frame, end_frame, first_tq, load_gt, train, mesh_gt)
        if slam:
            self.slam = SLAM(setting_file='./external/Uni-Fusion-use-ORB-SLAM2/Examples/RGB-D/scannet.yaml')
            if not load_gt:
                self.slam.feed_stack_and_start_thread(self.rgb, self.depth)
            else:
                # for rerun the predicted trajectory
                # load from predicted trajectory 'pred_traj.txt'
                frame_ids, poses = read_orbslam2_file(path+'/pred_traj.txt')
                # fill in not listed poses
                now_id = 0
                frame_ids_ = []
                poses_ = []
                for idx in range(0,frame_ids[-1]+1):
                    frame_ids_.append(idx)
                    if idx in frame_ids:
                        poses_.append(poses[now_id])
                        now_id += 1
                    else:
                        poses_.append(np.eye(4))
                frame_ids = frame_ids_
                self.slam.poses = poses_
        
        if True: # for evaluation only
            self.load_poses(path+'./pose')
            from pyquaternion import Quaternion
            with open(path+'/gt_traj.txt', 'w') as f:
                for idx, pose in enumerate(self.poses):
                    try:
                        q = Quaternion(matrix=pose[:3,:3], atol=1e-5, rtol=1e-5)
                    except Exception as e:
                        continue
                    f.write('%s %f %f %f %f %f %f %f\n'\
                                %(idx,\
                                pose[0,3], pose[1,3], pose[2,3],\
                                q[1], q[2], q[3], q[0]\
                                ))
             




        self.has_ir = has_ir
        self.has_saliency = has_saliency
        self.has_style = has_style
        self.has_latent = has_latent
        # saliency
        if has_saliency:
            from transparent_background import Remover 
            self.saliency_detector = Remover()

        # style
        if has_style:
            from external.style_transfer.experiments import style_api
            self.style_painting = style_api.get_api(style_idx)

        # latent
        self.latent_func = f_im

        # np_str
        if has_latent:
            self.np_image_strings = []
            for rgb_id in tqdm(self.rgb_ids):
                with tf.gfile.GFile(rgb_id, 'rb') as f:
                    np_image_string = np.array([f.read()])
                    self.np_image_strings.append(np_image_string)

        self.first_iso = motion_util.Isometry(q=Quaternion(array=[0.0, -1.0, 0.0, 0.0]))
        self.change_iso = None
    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(glob.glob(os.path.join(path, '*.txt')),
                            key=lambda x: int(os.path.basename(x)[:-4]))
        for id, pose_path in enumerate(pose_paths):
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                l = list(map(float, line.strip().split(' ')))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4)
            '''
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            '''
            #c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)


    def orbslam2pose_to_LIMpose(self, pose):
        try:
            cur_iso = motion_util.Isometry(q=Quaternion(matrix=pose[:3,:3], atol=1e-5, rtol=1e-5), t=pose[:3,3])
        except Exception as e:
            print(pose, e)
            return None
        if self.change_iso is None:
            self.change_iso = self.first_iso.dot(cur_iso.inv())
            self.T_gt2uni = self.change_iso.matrix
        return self.change_iso.dot(cur_iso)
    def __getitem__(self, idx): 

        frame_data = FrameData()
        if hasattr(self,'slam'):
            # wait slam to track
            while idx >= len(self.slam.poses):
                sleep(.1)
            frame_data.gt_pose = self.orbslam2pose_to_LIMpose(self.slam.poses[idx])
        else:
            if self.gt_trajectory is not None:
                frame_data.gt_pose = self.gt_trajectory[idx]
            else:
                frame_data.gt_pose = None


        frame_data.calib = FrameIntrinsic(577.590698, 578.729797, 318.905426, 242.683609, 1000.)
        frame_data.depth =  torch.from_numpy(self.depth[idx].astype(np.float32)).cuda(0).float() / 1000
        frame_data.rgb = torch.from_numpy(self.rgb[idx]).cuda(0).float() / 255.

        img = Image.fromarray((frame_data.rgb.cpu().numpy()*255).astype(np.ubyte)).convert('RGB')

        frame_data.ir = None
        frame_data.saliency = torch.from_numpy(
                self.saliency_detector.process(img,type='map').astype(np.float32)
                ).cuda(0).float() / 255. if self.has_saliency else None

        frame_data.style = torch.from_numpy(
                self.style_painting(img)).cuda(0).float() / 255. if self.has_style else None


        H,W,_ = frame_data.rgb.shape

        frame_data.latent = torch.from_numpy(self.latent_func(self.np_image_strings[idx], H, W)).cuda(0).float() if self.has_latent else None


        frame_data.customs = [frame_data.ir, frame_data.saliency, frame_data.style, frame_data.latent]
        return frame_data
