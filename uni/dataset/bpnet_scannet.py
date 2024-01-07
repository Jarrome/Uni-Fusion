import numpy as np
import cv2
import os
import time
import torch
import math

from collections import defaultdict, namedtuple

from threading import Thread, Lock
from dataset.production import *
from utils import motion_util
from pyquaternion import Quaternion
import tensorflow.compat.v1 as tf


import open3d as o3d

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





def make_pair(matrix, threshold=1):
    assert (matrix >= 0).all()
    pairs = []
    base = defaultdict(int)
    while True:
        i = matrix[:, 0].argmin()
        min0 = matrix[i, 0]
        j = matrix[0, :].argmin()
        min1 = matrix[0, j]

        if min0 < min1:
            i, j = i, 0
        else:
            i, j = 0, j
        if min(min1, min0) < threshold:
            pairs.append((i + base['i'], j + base['j']))

        matrix = matrix[i + 1:, j + 1:]
        base['i'] += (i + 1)
        base['j'] += (j + 1)

        if min(matrix.shape) == 0:
            break
    return pairs
# create camera intrinsics
def make_intrinsic(fx, fy, mx, my):
    intrinsic = np.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic


# create camera intrinsics
def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic


class ScanNetLatentDataset(object):

    def __init__(self, path, f_im, start_frame: int = 0, end_frame: int = -1, first_tq: list = None, load_gt: bool = False, register=True, mesh_gt: str = None):
        path = os.path.expanduser(path)

        fx=577.870605 
        fy=577.870605 
        mx=319.5
        my=239.5
        image_dim=(320, 240)
        self.intricsic = make_intrinsic(fx=fx, fy=fy, mx=mx, my=my)
        self.intricsic = adjust_intrinsic(self.intricsic, intrinsic_image_dim=[640, 480], image_dim=image_dim)
        self.calib = [self.intricsic[0,0], self.intricsic[1,1], self.intricsic[0,2], self.intricsic[1,2], 1000.]
        
        self.f_im = f_im

        self.first_iso = motion_util.Isometry(q=Quaternion(array=[0.0, -1.0, 0.0, 0.0]))


        if not register:
            rgb_ids, rgb_timestamps = self.listdir(path, 'color', ext='.jpg')
            depth_ids, depth_timestamps = self.listdir(path, 'depth')
        else:
            rgb_imgs, rgb_timestamps = self.listdir(path, 'color', ext='.jpg')
            depth_imgs, depth_timestamps = self.listdir(path, 'depth')
            
            interval = (rgb_timestamps[1:] - rgb_timestamps[:-1]).mean() * 2/3
            matrix = np.abs(rgb_timestamps[:, np.newaxis] - depth_timestamps)
            pairs = make_pair(matrix, interval)

            rgb_ids = []
            depth_ids = []
            for i, j in pairs:
                rgb_ids.append(rgb_imgs[i])
                depth_ids.append(depth_imgs[j])


        if load_gt:
            self.gt_trajectory = self._parse_traj_file(rgb_imgs)
            self.gt_trajectory = self.gt_trajectory[start_frame:end_frame]
            change_iso = self.first_iso.dot(self.gt_trajectory[0].inv())

            # remove inf pose in trajectory
            if len(self.untracked_ids) > 0:
                for id in self.untracked_ids[::-1]:
                    self.gt_trajectory.pop(id)
                    rgb_ids.pop(id)
                    depth_ids.pop(id)
 





            self.gt_trajectory = [change_iso.dot(t) for t in self.gt_trajectory]
            #assert len(self.gt_trajectory) == len(self.rgb)
            self.T_gt2uni = change_iso.matrix
        
        else:
            self.gt_trajectory = None
            self.T_gt2uni = self.first_iso.matrix


        self.rgb = ImageReader(rgb_ids, rgb_timestamps, is_rgb=True)
        self.np_image_strings = []
        for rgb_id in rgb_ids:
            '''
            rgb_id_ = rgb_id[39:]
            rgb_id_ = '/home/yijun/data/scannet/data/scans/'+rgb_id_
            '''
            with tf.gfile.GFile(rgb_id, 'rb') as f:
                np_image_string = np.array([f.read()])
                self.np_image_strings.append(np_image_string)
        self.depth = ImageReader(depth_ids, depth_timestamps)
        self.timestamps = rgb_timestamps

        self.frame_id = 0

        if mesh_gt is None:
            print("using reconstruction mesh")
        else:
            if mesh_gt != '' and load_gt:
                self.gt_mesh = self.get_ground_truth_mesh(mesh_gt)#, gt_traj_path)

    def _parse_traj_file(self,rgb_imgs):
        camera_ext = {}
        traj_data = []
        for i, rgb_name in enumerate(rgb_imgs):
            pose_name = rgb_name.replace('jpg','txt').replace('color','pose')
            pose = np.genfromtxt(pose_name)
            traj_data.append(pose)
        #cano_quat = motion_util.Isometry(q=Quaternion(axis=[0.0, 0.0, 1.0], degrees=180.0))
    
        self.untracked_ids = []
        for id, cur_p in enumerate(traj_data):
            if np.isinf(cur_p).sum()>0:
                print('Find untracked: Frame %d'%id)
                #T = traj_data[0].reshape((4,4))#.dot(z_back2forward)
                camera_ext[id] = None
                self.untracked_ids.append(id)
                continue


            T = cur_p.reshape((4,4))
            #cur_q = Quaternion(imaginary=cur_p[4:7], real=cur_p[-1]).rotation_matrix
            cur_q = T[:3,:3]
            cur_t = T[:3, 3]
            #print(T)
            #cur_iso = motion_util.Isometry(q=Quaternion(matrix=cur_q), t=cur_t)
            try:
                cur_iso = motion_util.Isometry(q=Quaternion(matrix=cur_q, atol=1e-5, rtol=1e-5), t=cur_t)
            except Exception as e:
                print(e)
                pdb.set_trace()
                print(e)

            camera_ext[id] = cur_iso #cano_quat.dot(cur_iso)
        camera_ext[len(camera_ext)] = camera_ext[len(camera_ext)-1]
        return [camera_ext[t] for t in range(len(camera_ext))]



    def get_ground_truth_PC(self, X):
        change_mat = self.T_gt2uni


        X_t = (change_mat[:3,:3].dot(X.T)+change_mat[:3,(3,)]).T
        return X_t



    def sort(self, xs, st = 3):
        return sorted(xs, key=lambda x:float(x[st:-4]))

    def listdir(self, path, split='rgb', ext='.png'):
        imgs, timestamps = [], []
        files = [x for x in os.listdir(os.path.join(path, split)) if x.endswith(ext)]
        st = 0
        for name in self.sort(files,st):
            imgs.append(os.path.join(path, split, name))
            timestamp = float(name[st:-len(ext)].rstrip('.'))
            timestamps.append(timestamp)

        return imgs, np.array(timestamps)

    def __getitem__(self, idx):
        frame_data = FrameData()
        if self.gt_trajectory is not None:
            frame_data.gt_pose = self.gt_trajectory[idx]
        else:
            frame_data.gt_pose = None
        frame_data.calib = FrameIntrinsic(*self.calib)
        frame_data.depth =  torch.from_numpy(self.depth[idx].astype(np.float32)).cuda(0).float() / 1000
        frame_data.rgb = torch.from_numpy(self.rgb[idx]).cuda(0).float() / 255.
        # lseg
        #frame_data.latent = self.f_im(frame_data.rgb).permute(2,3,1,0).squeeze(-1)
        # openseg
        #if idx == 49:
        #    pdb.set_trace()
        H,W,_ = frame_data.rgb.shape

        frame_data.latent = torch.from_numpy(self.f_im(self.np_image_strings[idx], H, W)).cuda(0).float()


        return frame_data
    def __next__(self):
        frame_data = FrameData()
        if self.gt_trajectory is not None:
            frame_data.gt_pose = self.gt_trajectory[self.frame_id]
        else:
            frame_data.gt_pose = None
        frame_data.calib = FrameIntrinsic(*self.calib)

        H,W = elf.depth[self.frame_id].shape
        frame_data.depth =  torch.from_numpy(self.depth[self.frame_id].astype(np.float32)).cuda().float() / 1000
        frame_data.rgb = torch.from_numpy(self.rgb[self.frame_id]).cuda().float() / 255.
        # lseg
        #frame_data.latent = self.f_im(frame_data.rgb).permute(2,3,1,0).squeeze(-1)
        # openseg
        frame_data.latent = torch.from_numpy(self.f_im(self.np_image_strings[self.frame_id])).cuda().float()

        self.frame_id += 1
        return frame_data

    def __len__(self):
        return len(self.rgb)


