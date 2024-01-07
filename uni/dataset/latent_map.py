import numpy as np
import cv2
import os
import time
from math import *
import torch

from threading import Thread, Lock

from collections import defaultdict, namedtuple

from dataset.production import NICE_SLAM_dataset, FrameData, FrameIntrinsic

from utils import motion_util
from pyquaternion import Quaternion

from utils.exp_util import parse_config_yaml
from pathlib import Path
import argparse
import tensorflow.compat.v1 as tf

from scipy.spatial.transform import Rotation



import open3d as o3d
import json

import pdb


class ImageReader(object):
    def __init__(self, ids, timestamps=None, cam=None, is_rgb=False):
        self.ids = ids
        self.timestamps = timestamps
        self.cam = cam
        self.cache = dict()
        self.idx = 0

        self.is_rgb = is_rgb

        self.ahead = 10      # 10 images ahead of current index
        self.waiting = 1.5   # waiting time

        self.preload_thread = Thread(target=self.preload)
        self.thread_started = False

    def read(self, path):
        img = cv2.imread(path, -1)
        if self.is_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.cam is None:
            return img
        else:
            return self.cam.rectify(img)
        
    def preload(self):
        idx = self.idx
        t = float('inf')
        while True:
            if time.time() - t > self.waiting:
                return
            if self.idx == idx:
                time.sleep(1e-2)
                continue
            
            for i in range(self.idx, self.idx + self.ahead):
                if i not in self.cache and i < len(self.ids):
                    self.cache[i] = self.read(self.ids[i])
            if self.idx + self.ahead > len(self.ids):
                return
            idx = self.idx
            t = time.time()
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        self.idx = idx
        # if not self.thread_started:
        #     self.thread_started = True
        #     self.preload_thread.start()

        if idx in self.cache:
            img = self.cache[idx]
            del self.cache[idx]
        else:   
            img = self.read(self.ids[idx])
                    
        return img

    def __iter__(self):
        for i, timestamp in enumerate(self.timestamps):
            yield timestamp, self[i]

    @property
    def dtype(self):
        return self[0].dtype
    @property
    def shape(self):
        return self[0].shape





#img, depth = read_files(basedir, frame['file_path'], frame['depth_file_path'])

def read_files(basedir, rgb_file, depth_file):
    fname = os.path.join(basedir, rgb_file)
    img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    if img.shape[-1] == 4:
        convert_fn = cv2.COLOR_BGRA2RGBA
    else:
        convert_fn = cv2.COLOR_BGR2RGB
    img = (cv2.cvtColor(img, convert_fn) / 255.).astype(np.float32) # keep 4 channels (RGBA) if available
    depth_fname = os.path.join(basedir, depth_file)
    depth = cv2.imread(depth_fname, cv2.IMREAD_UNCHANGED).astype(np.float64)
    return img, depth




class TwoDThreeDSDataset(object):
    '''
    '''

    def __init__(self, path, start_frame: int = 0, end_frame: int = -1, first_tq: list = None, load_gt: bool = False, train=True, mesh_gt = None):

        self.first_iso = motion_util.Isometry(q=Quaternion(array=[0.0, -1.0, 0.0, 0.0]))


        rgb_ids, rgb_timestamps = self.listdir(path, 'rgb')
        depth_ids = [rgb_id.replace('rgb','depth') for rgb_id in rgb_ids]
        depth_timestamps = rgb_timestamps

        self.rgb = ImageReader(rgb_ids, rgb_timestamps, is_rgb=True)
        self.depth = ImageReader(depth_ids, depth_timestamps)


        pose_ids = [rgb_id.replace('rgb','pose').replace('png','json') for rgb_id in rgb_ids]
        
        self.np_image_strings = []
        for rgb_id in rgb_ids:
            with tf.gfile.GFile(rgb_id, 'rb') as f:
                np_image_string = np.array([f.read()])
                self.np_image_strings.append(np_image_string)


        
        if load_gt:

            self.gt_trajectory, self.intrins = self._parse_traj_file(pose_ids)
            self.gt_trajectory = self.gt_trajectory[start_frame:end_frame]
            change_iso = self.first_iso.dot(self.gt_trajectory[0].inv())
            self.gt_trajectory = [change_iso.dot(t) for t in self.gt_trajectory]
            #assert len(self.gt_trajectory) == len(self.rgb)
            self.T_gt2uni = change_iso.matrix

        
        else:
            self.gt_trajectory = None
            self.T_gt2uni = self.first_iso.matrix

        self.frame_id = 0

 
    def __len__(self):
        return len(self.rgb)


    def get_ground_truth_PC(self, X):
        change_mat = self.T_gt2uni


        X_t = (change_mat[:3,:3].dot(X.T)+change_mat[:3,(3,)]).T
        return X_t


    def _parse_traj_file(self,pose_files):
        camera_ext = {}
        poses = []
        intrins = []
        for pose_file in pose_files:
            with open(pose_file, 'r') as f:
                pose_dict = json.load(f)
                pose = np.eye(4)
                pose[:3,:] = np.array(pose_dict['camera_rt_matrix'])
                pose = np.linalg.inv(pose)

                intrin = np.array(pose_dict['camera_k_matrix'])

                poses.append(pose)
                intrins.append(intrin)

        #cano_quat = motion_util.Isometry(q=Quaternion(axis=[0.0, 0.0, 1.0], degrees=180.0))
        for id, cur_p in enumerate(poses):
            T = cur_p.reshape((4,4))
            #cur_q = Quaternion(imaginary=cur_p[4:7], real=cur_p[-1]).rotation_matrix
            cur_q = T[:3,:3]
            cur_t = T[:3, 3]
            #cur_q[1] = -cur_q[1]
            #cur_q[:, 1] = -cur_q[:, 1]
            #cur_t[1] = -cur_t[1]
            cur_iso = motion_util.Isometry(q=Quaternion(matrix=cur_q, atol=1e-5, rtol=1e-5), t=cur_t)
            camera_ext[id] = cur_iso #cano_quat.dot(cur_iso)
        camera_ext[len(camera_ext)] = camera_ext[len(camera_ext)-1]
        return [camera_ext[t] for t in range(len(camera_ext))], intrins



    def listdir(self, path, split='rgb', ext='.png'):
        imgs, timestamps = [], []
        files = [x for x in os.listdir(os.path.join(path, split)) if x.endswith(ext)]
        
        for name in files:
            imgs.append(os.path.join(path, split, name))

        return imgs, np.arange(len(imgs))




class ScanNetDataset(object):
    '''
    '''

    def __init__(self, path, start_frame: int = 0, end_frame: int = -1, first_tq: list = None, load_gt: bool = False, train=True, mesh_gt = None):

        cfg_path = "./dataset/production/NICE_SLAM_config/demo.yaml"
        cfg_ns = parse_config_yaml(Path(cfg_path))
        cfg = {}
        cfg['cam'] = cfg_ns.cam
        cfg['dataset'] = cfg_ns.dataset
        cfg['data'] = cfg_ns.data

        args = argparse.Namespace()
        args.input_folder = path
        self.inner_dataset = NICE_SLAM_dataset.ScanNet(cfg, args, scale=1)
        
        path = os.path.expanduser(path)
        self.first_iso = motion_util.Isometry(q=Quaternion(array=[0.0, -1.0, 0.0, 0.0]))


        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        self.depth_scaling_factor = cfg['cam']['png_depth_scale']


        
        if load_gt:

            self.gt_trajectory = self._parse_traj_file()
            self.gt_trajectory = self.gt_trajectory[start_frame:end_frame]
            change_iso = self.first_iso.dot(self.gt_trajectory[0].inv())

            # remove inf pose in trajectory
            if len(self.untracked_ids) > 0:
                for id in self.untracked_ids[::-1]:
                    self.gt_trajectory.pop(id)
                    self.inner_dataset.color_paths.pop(id)
                    self.inner_dataset.depth_paths.pop(id)
                    self.inner_dataset.poses.pop(id)
                self.inner_dataset.n_img = len(self.inner_dataset.color_paths)

            self.gt_trajectory = [change_iso.dot(t) for t in self.gt_trajectory]
            #assert len(self.gt_trajectory) == len(self.rgb)
            self.T_gt2uni = change_iso.matrix

        
        else:
            self.gt_trajectory = None
            self.T_gt2uni = self.first_iso.matrix


        '''
        if mesh_gt is "":
            print("using reconstruction mesh")
        else:
            self.gt_mesh = self.get_ground_truth_mesh(mesh_gt)
            #o3d.io.write_triangle_mesh('tmp_.ply', self.gt_mesh)
        '''

        self.timestamps = list(range(len(self.inner_dataset)))



        self.frame_id = 0

    def get_ground_truth_PC(self, X):
        '''
        z_back2forward = np.eye(4)
        z_back2forward[1,1] = -1
        z_back2forward[2,2] = -1
        T0 = self.inner_dataset.poses[0].numpy().reshape((4,4))
        T0 = T0.dot(z_back2forward)
        change_mat = (self.first_iso.matrix.dot(np.linalg.inv(T0)))
        '''
        change_mat = self.T_gt2uni


        X_t = (change_mat[:3,:3].dot(X.T)+change_mat[:3,(3,)]).T
        return X_t

    def get_ground_truth_mesh(self, mesh_path):
        import trimesh

        '''
        z_back2forward = np.eye(4)
        z_back2forward[1,1] = -1
        z_back2forward[2,2] = -1
        T0 = self.inner_dataset.poses[0].numpy().reshape((4,4))
        T0 = T0.dot(z_back2forward)
        change_mat = (self.first_iso.matrix.dot(np.linalg.inv(T0)))
        '''

        '''
        axisAlignment = '-0.043619 0.999048 0.000000 -2.248810 -0.999048 -0.043619 0.000000 2.434610 0.000000 0.000000 1.000000 -0.206373 0.000000 0.000000 0.000000 1.000000'.split(' ')
        axisAlignment = np.array([float(item) for item in axisAlignment])
        axisAlignment = axisAlignment.reshape(4,4)
        '''

        change_mat = self.T_gt2uni

        #self.T_gt2uni = change_mat
        mesh_gt = trimesh.load(mesh_path)
        #mesh_gt.apply_transform(axisAlignment)
        #mesh_gt.apply_transform(axisAlignment)

        mesh_gt.apply_transform(change_mat)
        return mesh_gt.as_open3d





    def _parse_traj_file(self):
        '''
        z_back2forward = np.eye(4)
        z_back2forward[1,1] = -1
        z_back2forward[2,2] = -1
        '''
        camera_ext = {}
        traj_data = [pose.numpy() for pose in self.inner_dataset.poses]
        #cano_quat = motion_util.Isometry(q=Quaternion(axis=[0.0, 0.0, 1.0], degrees=180.0))

        self.untracked_ids = []

        for id, cur_p in enumerate(traj_data):
            if np.isinf(cur_p).sum()>0:
                print('Find untracked: Frame %d'%id)
                #T = traj_data[0].reshape((4,4))#.dot(z_back2forward)
                camera_ext[id] = None
                self.untracked_ids.append(id)
            else:
                T = cur_p.reshape((4,4))#.dot(z_back2forward)
                cur_q = T[:3,:3]
                cur_t = T[:3,3]
                #cur_iso = motion_util.Isometry.from_matrix(T)#,ortho=True)#(q=Quaternion(matrix=cur_q, atol=1e-1, rtol=1e-1), t=cur_t)
                cur_iso = motion_util.Isometry(q=Quaternion(matrix=cur_q, atol=1e-5, rtol=1e-5), t=cur_t)
                camera_ext[id] = cur_iso #cano_quat.dot(cur_iso)
        camera_ext[len(camera_ext)] = camera_ext[len(camera_ext)-1]
        return [camera_ext[t] for t in range(len(camera_ext))]

    def __getitem__(self, idx):
        index, rgb, depth, pose = self.inner_dataset[idx]

        frame_data = FrameData()
        frame_data.calib = FrameIntrinsic(self.fx, self.fy, self.cx, self.cy, self.depth_scaling_factor)
        frame_data.depth = depth.float()# torch.from_numpy(self.depth[idx_id].astype(np.float32)).cuda().float()# / self.depth_scaling_factor
        frame_data.rgb = rgb.float() #torch.from_numpy(self.rgb[idx_id]).cuda().float() #/ 255.

        frame_data.gt_pose = self.gt_trajectory[idx] 
        return frame_data



    def __next__(self):
        index, rgb, depth, pose = self.inner_dataset[self.frame_id]

        frame_data = FrameData()
        frame_data.calib = FrameIntrinsic(self.fx, self.fy, self.cx, self.cy, self.depth_scaling_factor)
        frame_data.depth = depth.float()# torch.from_numpy(self.depth[idx_id].astype(np.float32)).cuda().float()# / self.depth_scaling_factor
        frame_data.rgb = rgb.float() #torch.from_numpy(self.rgb[idx_id]).cuda().float() #/ 255.

        frame_data.gt_pose = self.gt_trajectory[self.frame_id] 



        self.frame_id += 1
        return frame_data

    def __len__(self):
        return len(self.inner_dataset)




class ScanNetLatentDataset(ScanNetDataset):
    '''
    '''

    def __init__(self, path, f_im, start_frame: int = 0, end_frame: int = -1, first_tq: list = None, load_gt: bool = False, train=True, mesh_gt = None):
        super(ScanNetLatentDataset, self).__init__(path, start_frame, end_frame, first_tq, load_gt, train, mesh_gt)
        self.f_im = f_im
    
    def __getitem__(self, idx):
        index, rgb, depth, pose = self.inner_dataset[idx]

        frame_data = FrameData()
        frame_data.calib = FrameIntrinsic(self.fx, self.fy, self.cx, self.cy, self.depth_scaling_factor)
        frame_data.depth = depth.float()# torch.from_numpy(self.depth[idx_id].astype(np.float32)).cuda().float()# / self.depth_scaling_factor
        frame_data.rgb = rgb.float() #torch.from_numpy(self.rgb[idx_id]).cuda().float() #/ 255.
        frame_data.gt_pose = self.gt_trajectory[idx] 
        frame_data.latent = self.f_im(frame_data.rgb).permute(2,3,1,0).squeeze(-1)

        return frame_data
    def __next__(self):

        index, rgb, depth, pose = self.inner_dataset[self.frame_id]

        frame_data = FrameData()
        frame_data.calib = FrameIntrinsic(self.fx, self.fy, self.cx, self.cy, self.depth_scaling_factor)
        frame_data.depth = depth.float()# torch.from_numpy(self.depth[idx_id].astype(np.float32)).cuda().float()# / self.depth_scaling_factor
        frame_data.rgb = rgb.float() #torch.from_numpy(self.rgb[idx_id]).cuda().float() #/ 255.
        frame_data.gt_pose = self.gt_trajectory[self.frame_id] 
        frame_data.latent = self.f_im(frame_data.rgb).permute(2,3,1,0).squeeze(-1)

        self.frame_id += 1
        return frame_data

class TwoDThreeDSLatentDataset(TwoDThreeDSDataset):
    def __init__(self, path, f_im, start_frame: int = 0, end_frame: int = -1, first_tq: list = None, load_gt: bool = False, train=True, mesh_gt = None):
        super(TwoDThreeDSLatentDataset, self).__init__(path, start_frame, end_frame, first_tq, load_gt, train, mesh_gt)
        self.f_im = f_im
    
    def __getitem__(self, idx, cuda_id=0):
        frame_data = FrameData()

        intrin = self.intrins[idx]

        # resize to 640
        H,W = self.depth[idx].shape
        assert H==W, 'H==W'
        H_s = 640
        depth = cv2.resize(self.depth[idx], (H_s,H_s), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.resize(self.rgb[idx], (H_s,H_s), interpolation=cv2.INTER_LINEAR)
        ratio = H_s / H

        frame_data.calib = FrameIntrinsic(intrin[0,0] * ratio, intrin[1,1]* ratio, intrin[0,2]* ratio, intrin[1,2]* ratio, 512.)
        frame_data.depth = torch.from_numpy(depth.astype(np.float32)).cuda(cuda_id).float() / 512. 
        frame_data.rgb = torch.from_numpy(rgb).cuda(cuda_id).float() / 255.
        frame_data.gt_pose = self.gt_trajectory[idx] 

        H,W,_ = frame_data.rgb.shape

        frame_data.latent =  torch.from_numpy(self.f_im(self.np_image_strings[idx],H,W)).cuda(cuda_id).float()
        return frame_data

