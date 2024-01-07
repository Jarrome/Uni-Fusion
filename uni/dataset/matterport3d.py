import numpy as np
import cv2
import os
import time
import torch

from collections import defaultdict, namedtuple

from threading import Thread, Lock
from dataset.production import *
from utils import motion_util
from pyquaternion import Quaternion

import open3d as o3d

import pdb




class Matterport3DRGBDDataset():
    '''
        follow https://github.com/otakuxiang/circle/blob/master/torch/sample_matterport.py
    '''

    def __init__(self, path, start_frame: int = 0, end_frame: int = -1, first_tq: list = None, load_gt: bool = False, mesh_gt: str = None):
        path = os.path.expanduser(path)

        self.first_iso = motion_util.Isometry(q=Quaternion(array=[0.0, -1.0, 0.0, 0.0]))

        self.depth_path = os.path.join(path,"matterport_depth_images")
        self.rgb_path = os.path.join(path,"matterport_color_images")
        self.pose_path = os.path.join(path,"matterport_camera_poses")
        self.intri_path = os.path.join(path,"matterport_camera_intrinsics")
        tripod_numbers = [ins[:ins.find("_")] for ins in os.listdir(self.intri_path)]
        self.depthMapFactor = 4000


        self.frames = []
        for tripod_number in tripod_numbers:
            for camera_id in range(3):
                for frame_id in range(6):
                    self.frames.append([tripod_number,camera_id,frame_id])
        self.frame_ids = list(range(len(self.frames)))


        if load_gt:
            self.gt_trajectory = self._parse_traj_file(self.pose_path)
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
        return len(self.frames)

    def _parse_traj_file(self, traj_path):
        traj_data = []
        for frame_id in range(len(self)):
            tripod_number,camera_id,frame_idx = self.frames[frame_id]
            f = open(os.path.join(self.pose_path,f"{tripod_number}_pose_{camera_id}_{frame_idx}.txt"))
            pose = np.zeros((4,4))
            for idx,line in enumerate(f):
                ss = line.strip().split(" ")
                for k in range(0,4):
                    pose[idx,k] = float(ss[k])
            # pose = np.linalg.inv(pose)
            traj_data.append(pose)

            f.close()
        
        camera_ext = {}
        for id, cur_p in enumerate(traj_data):
            T = cur_p
            cur_q = T[:3,:3]
            cur_t = T[:3, 3]
            cur_iso = motion_util.Isometry(q=Quaternion(matrix=cur_q, atol=1e-5, rtol=1e-5), t=cur_t)
            camera_ext[id] = cur_iso
        camera_ext[len(camera_ext)] = camera_ext[len(camera_ext)-1]
        return [camera_ext[t] for t in range(len(camera_ext))]




    
    def __getitem__(self, frame_id):
        tripod_number,camera_id,frame_idx = self.frames[frame_id]
        '''
        f = open(os.path.join(self.pose_path,f"{tripod_number}_pose_{camera_id}_{frame_idx}.txt"))
        pose = np.zeros((4,4))
        for idx,line in enumerate(f):
            ss = line.strip().split(" ")
            for k in range(0,4):
                pose[idx,k] = float(ss[k])
        # pose = np.linalg.inv(pose)
        pose = torch.from_numpy(pose).float()
        
        f.close()
        '''
        K_depth = np.zeros((3,3))
        f = open(os.path.join(self.intri_path,f"{tripod_number}_intrinsics_{camera_id}.txt"))
        p = np.zeros((4))
        for idx,line in enumerate(f):
            ss = line.strip().split(" ")   
            for j in range(4):
                p[j] = float(ss[j+2])
        f.close()
        K_depth[0,0] = p[0]
        K_depth[1,1] = p[1]
        K_depth[2,2] = 1
        K_depth[0,2] = p[2]
        K_depth[1,2] = p[3]
        depth_path = os.path.join(self.depth_path,tripod_number+"_d"+str(camera_id)+"_"+str(frame_idx)+".png")
        depth =cv2.imread(depth_path,-1)
        rgb_path = os.path.join(self.rgb_path,tripod_number+"_i"+str(camera_id)+"_"+str(frame_idx)+".jpg")
        rgb = cv2.imread(rgb_path, -1)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        ins = torch.from_numpy(K_depth).float()
        if depth is None:
            print("get None image!")
            print(depth_path)
            return None
        
        depth = depth.astype(np.float32) / self.depthMapFactor
        depth = torch.from_numpy(depth).float()

        rgb = rgb.astype(np.float32) / 255
        rgb = torch.from_numpy(rgb).float()

        assert depth.shape[:2] == rgb.shape[:2], 'depth shape should == rgb shape'
        
        return rgb,depth,ins
    def __next__(self):
        rgb, depth, K = self[self.frame_id]

        frame_data = FrameData()
        frame_data.calib = FrameIntrinsic(K[0,0],K[1,1],K[0,2],K[1,2],self.depthMapFactor)
        frame_data.depth = depth.cuda()
        frame_data.rgb = rgb.cuda()
        frame_data.gt_pose = self.gt_trajectory[self.frame_id]

        self.frame_id += 1
        return frame_data


