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







class TUMRGBDDataset():
    def __init__(self, path, start_frame: int = 0, end_frame: int = -1, first_tq: list = None, load_gt: bool = False, register=True, mesh_gt: str = None):
        path = os.path.expanduser(path)
        self.first_iso = motion_util.Isometry(q=Quaternion(array=[0.0, -1.0, 0.0, 0.0]))
        rgb_ids = []
        depth_ids = []
        self.timestamps = []
        with open(os.path.join(path,'asso.txt'),'r') as f:
            ls = f.readlines()
        for l in ls:
            elems = l.strip().split(' ')
            rgb_id = elems[1]
            depth_id = elems[3]
            timestamp = elems[0]
            rgb_ids.append(os.path.join(path,rgb_id))
            depth_ids.append(os.path.join(path,depth_id))
            self.timestamps.append(timestamp)


        
        self.rgb = ImageReader(rgb_ids)
        self.depth = ImageReader(depth_ids)

        self.frame_id = 0



        assert load_gt == False, "NO TUM GT TRAJECTORY"
        self.gt_trajectory = None
        self.T_gt2uni = self.first_iso.matrix




    def sort(self, xs):
        return sorted(xs, key=lambda x:float(x[:-4]))

    def __getitem__(self, idx):
        frame_data = FrameData()
        frame_data.gt_pose = None
        frame_data.calib = FrameIntrinsic(525., 525., 319.5, 239.5, 5000)
        frame_data.depth =  torch.from_numpy(self.depth[idx].astype(np.float32)).cuda().float() / 5000
        frame_data.rgb = torch.from_numpy(self.rgb[idx]).cuda().float() / 255.
        return frame_data


    def __next__(self):
        frame_data = FrameData()
        frame_data.gt_pose = None
        frame_data.calib = FrameIntrinsic(525., 525., 319.5, 239.5, 5000)
        frame_data.depth =  torch.from_numpy(self.depth[self.frame_id].astype(np.float32)).cuda().float() / 5000
        frame_data.rgb = torch.from_numpy(self.rgb[self.frame_id]).cuda().float() / 255.
        self.frame_id += 1
        return frame_data



    def __len__(self):
        return len(self.rgb)
