import numpy as np
import torch
import cv2
import yaml
import argparse
from pyquaternion import Quaternion
from uni.encoder.uni_encoder_v2 import get_uni_model
from uni.mapper.surface_map import SurfaceMap
from uni.mapper.context_map_v2 import ContextMap # 8 points
import uni.tracker.tracker_custom as tracker

from uni.dataset import FrameIntrinsic
from uni.utils import motion_util

import pdb


def get_modules(main_device='cuda:0'):
    with open('configs/replica/office0.yaml') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    args = argparse.Namespace(**configs)

    args.surface_mapping = argparse.Namespace(**(args.surface_mapping))
    args.context_mapping = argparse.Namespace(**(args.context_mapping))
    args.tracking = argparse.Namespace(**(args.tracking))

    uni_model = get_uni_model(main_device)
    cm = ContextMap(uni_model,
                    args.context_mapping, 
                    uni_model.color_code_length,
                    device=main_device,
                    enable_async=False)

    sm = SurfaceMap(uni_model, 
                    cm,
                    args.surface_mapping, 
                    uni_model.surface_code_length, 
                    device=main_device,
                    enable_async=False)

    tk = tracker.SDFTracker(sm, args.tracking)

    return sm, cm, tk, args
       
def get_example_data(main_device='cuda:0'):

    colors, depths, poses = [], [], []
    for name_rgb, name_depth in [('example/office0/frame000000.jpg', 'example/office0/depth000000.png'),
                                ('example/office0/frame000020.jpg', 'example/office0/depth000020.png')]:
        rgb = cv2.imread(name_rgb,-1)
        depth = cv2.imread(name_depth,-1)

        color = torch.from_numpy(rgb).to(main_device).float() / 255.
        depth = torch.from_numpy(depth.astype(np.float32)).to(main_device).float() / 6553.5
        
        colors.append(color)
        depths.append(depth)

           

    customs = [None] * 4
    calib = FrameIntrinsic(600., 600., 599.5, 339.5, 6553.5)

    first_iso = motion_util.Isometry(q=Quaternion(array=[0.0, -1.0, 0.0, 0.0]))
    traj_mat = np.genfromtxt('example/office0/traj.txt').reshape((-1,4,4))
    for i in [0,20]:
        T = traj_mat[i,:,:]
        pose = first_iso.dot(motion_util.Isometry.from_matrix(T))
        poses.append(pose)

    return colors, depths, customs, calib, poses
