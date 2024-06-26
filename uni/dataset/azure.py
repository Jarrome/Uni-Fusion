import numpy as np
import cv2
import os
import time
import torch

from collections import defaultdict, namedtuple

from threading import Thread, Lock
from uni.dataset import *
from uni.utils import motion_util
from pyquaternion import Quaternion

import open3d as o3d

from PIL import Image


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



def read_orbslam2_file(traj_file):
    with open(traj_file) as f:
        lines = f.readlines()
    poses = []
    frame_ids = []
    for line_id, line in enumerate(lines):
        vs = [float(v) for v in line.strip().split(' ')]
        frame_id = round(vs[0]*30)
        #frame_id = round(vs[0])
        v_t = vs[1:4]
        #v_q = vs[4:] # xyzw
        v_q = Quaternion(vs[-1],*vs[4:-1])
        pose = v_q.transformation_matrix
        pose[:3,3] = np.array(v_t)
        poses.append(pose)
        frame_ids.append(frame_id)
    return frame_ids, poses




class AzureRGBDIDataset(object):

    def __init__(self, path, start_frame: int = 0, end_frame: int = -1, first_tq: list = None, load_gt: bool = False, register=True, mesh_gt: str = None):
        path = os.path.expanduser(path)


        cam = np.genfromtxt(path+'/intrinsic.txt')
        self.cam = namedtuple('camera', 'fx fy cx cy scale')(
                *(cam.tolist()), 1000)

        self.first_iso = motion_util.Isometry(q=Quaternion(array=[0.0, -1.0, 0.0, 0.0]))


        self.start_frame = start_frame
        self.end_frame = end_frame

        rgb_ids, rgb_timestamps = self.listdir(path, 'color')
        depth_ids, depth_timestamps = self.listdir(path, 'depth')
        ir_ids, ir_timestamps = self.listdir(path, 'ir')


        if load_gt:
            traj_path = path+'/traj_orbslam2.txt'
            frame_ids, poses = read_orbslam2_file(traj_path)
            rgb_ids = [rgb_ids[frame_id] for frame_id in frame_ids]
            depth_ids = [depth_ids[frame_id] for frame_id in frame_ids]
            ir_ids = [ir_ids[frame_id] for frame_id in frame_ids]
            rgb_timestamps = [rgb_timestamps[frame_id] for frame_id in frame_ids]
            '''
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
            poses = poses_
            '''

            self.gt_trajectory = self._parse_traj(poses)
            self.gt_trajectory = self.gt_trajectory[start_frame:end_frame]
            change_iso = self.first_iso.dot(self.gt_trajectory[0].inv())
            self.gt_trajectory = [change_iso.dot(t) for t in self.gt_trajectory]
            self.T_gt2uni = change_iso.matrix

        else:
            self.gt_trajectory = None
            self.T_gt2uni = self.first_iso.matrix



        self.rgb_ids = rgb_ids
        self.rgb = ImageReader(rgb_ids, rgb_timestamps, is_rgb=True)
        self.depth = ImageReader(depth_ids, depth_timestamps)
        self.ir = ImageReader(ir_ids, ir_timestamps)
 
        self.timestamps = rgb_timestamps

        self.frame_id = 0

        if mesh_gt is None:
            print("using reconstruction mesh")
        else:
            if mesh_gt != '' and load_gt:
                self.gt_mesh = self.get_ground_truth_mesh(mesh_gt)#, gt_traj_path)

        # saliency
        from transparent_background import Remover 
        self.saliency_detector = Remover()

        # style
        from thirdparts.style_transfer.experiments import style_api
        self.style_painting = style_api.get_api()


    def _parse_traj(self,traj_data):
        camera_ext = {}
        for id, cur_p in enumerate(traj_data):
            T = cur_p.reshape((4,4))
            #cur_q = Quaternion(imaginary=cur_p[4:7], real=cur_p[-1]).rotation_matrix
            cur_q = T[:3,:3]
            cur_t = T[:3, 3]
            cur_iso = motion_util.Isometry(q=Quaternion(matrix=cur_q), t=cur_t)
            camera_ext[id] = cur_iso #cano_quat.dot(cur_iso)
        camera_ext[len(camera_ext)] = camera_ext[len(camera_ext)-1]
        return [camera_ext[t] for t in range(len(camera_ext))]





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

        imgs = imgs[self.start_frame: self.end_frame]
        timestamps = timestamps[self.start_frame: self.end_frame]

        return imgs, np.array(timestamps)

    def __getitem__(self, idx):
        frame_data = FrameData()
        if self.gt_trajectory is not None:
            frame_data.gt_pose = self.gt_trajectory[idx]
        else:
            frame_data.gt_pose = None
        frame_data.calib = FrameIntrinsic(self.cam.fx, self.cam.fy, self.cam.cx, self.cam.cy, self.cam.scale)
        frame_data.depth =  torch.from_numpy(self.depth[idx].astype(np.float32)).cuda().float() / self.cam.scale
        frame_data.rgb = torch.from_numpy(self.rgb[idx]).cuda().float() / 255.
        frame_data.ir = torch.from_numpy(self.ir[idx].astype(np.float32)).cuda().float().unsqueeze(-1)

        img = Image.fromarray(self.rgb[idx]).convert('RGB')
        frame_data.saliency = torch.from_numpy(
                self.saliency_detector.process(img,type='map')
                ).cuda().float() /255


        frame_data.style = torch.from_numpy(
                self.style_painting(img)).cuda().float() / 255.


        return frame_data


    def __next__(self):
        frame_data = FrameData()
        if self.gt_trajectory is not None:
            frame_data.gt_pose = self.gt_trajectory[self.frame_id]
        else:
            frame_data.gt_pose = None
        frame_data.calib = FrameIntrinsic(self.cam.fx, self.cam.fy, self.cam.cx, self.cam.cy, self.cam.scale)
        frame_data.depth =  torch.from_numpy(self.depth[self.frame_id].astype(np.float32)).cuda().float() / self.cam.scale
        frame_data.rgb = torch.from_numpy(self.rgb[self.frame_id]).cuda().float() / 255.
        frame_data.ir = torch.from_numpy(self.ir[self.frame_id].astype(np.float32)).cuda().float()

        img = Image.fromarray(frame).convert('RGB')
        frame_data.saliency = torch.from_numpy(
                self.saliency_detector.process(img,type='map').astype(np.float32)
                ).cuda().float()


        frame_data.style = torch.from_numpy(
                self.style_painting(img)).cuda().float() / 255.


        self.frame_id += 1
        return frame_data

    def __len__(self):
        return len(self.rgb)

