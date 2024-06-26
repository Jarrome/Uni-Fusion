import sys, os
import pathlib
import importlib
import open3d as o3d
import trimesh
import argparse
from pathlib import Path
import logging
from time import time
import torch

import cv2
import numpy as np
from tqdm import tqdm



p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(p)


from uni.encoder.uni_encoder_v2 import get_uni_model 
from uni.mapper.context_map_v2 import ContextMap
from uni.utils import exp_util, vis_util, motion_util
from pyquaternion import Quaternion
from uni.utils.ray_cast import RayCaster

import pdb




#from utils.ray_cast import RayCaster

def depth2pc(depth_im, calib_mat):
    H,W = depth_im.shape

    d = depth_im.reshape(-1)
    
    fx = calib_mat[0,0]
    fy = calib_mat[1,1]
    cx = calib_mat[0,2]
    cy = calib_mat[1,2]

    x = np.arange(W)
    y = np.arange(H)
    yv, xv = np.meshgrid(y, x, indexing='ij') # HxW

    yv = yv.reshape(-1) # HW
    xv = xv.reshape(-1) # HW

    pc = np.zeros((H*W,3))
    pc[:,0] = (xv - cx) / fx * d
    pc[:,1] = (yv - cy) / fy * d
    pc[:,2] = d
    return pc













if __name__ == '__main__':
    config_path = sys.argv[1]

    args = exp_util.parse_config_yaml(Path(config_path))
    mesh_path = args.outdir+'/final_recons.ply'
    render_path = args.outdir+'/render/'


    if not hasattr(args,'slam'): 
        slamed = False
    else:
        slamed = args.slam
    use_gt = args.sequence_kwargs['load_gt'] and not slamed 

    if use_gt:
        traj_path = sys.argv[2]




    args.context_mapping = exp_util.dict_to_args(args.context_mapping)
    main_device = torch.device('cuda',index=0)
    uni_model = get_uni_model(main_device)
    context_map = ContextMap(uni_model, 
            args.context_mapping, uni_model.color_code_length, device=main_device,
                                        enable_async=args.run_async)

    context_map.load(args.outdir+'/color.lim')

    # Load in sequence.
    seq_package, seq_class = args.sequence_type.split(".")
    sequence_module = importlib.import_module("uni.dataset." + seq_package)
    sequence_module = getattr(sequence_module, seq_class)
    sequence = sequence_module(**args.sequence_kwargs)


    mesh = o3d.io.read_triangle_mesh(mesh_path)
    traj_data = np.genfromtxt(traj_path)
    
    render_path = pathlib.Path(render_path)
    render_path.mkdir(parents=True, exist_ok=True)


    calib_matrix = np.eye(3)
    calib_matrix[0,0] = 600.
    calib_matrix[1,1] = 600.
    calib_matrix[0,2] = 599.5
    calib_matrix[1,2] = 339.5

    H = 680
    W = 1200
    

    # if using gt_traj for trajectory, change_mat == I
    # else will need inv(traj_data[0])
    if use_gt:
        change_mat = np.eye(4)
    else:
        change_mat = np.linalg.inv(traj_data[0].reshape(4,4)) 

    ray_caster = RayCaster(mesh, H, W, calib_matrix)
    for id, pose in tqdm(enumerate(traj_data)):
        pose = pose.reshape((4,4))
        pose = change_mat.dot(pose)
        # 2. predict
        ans, ray_direction = ray_caster.ray_cast(pose) # N,3
        depth_on_ray = ans['t_hit'].numpy().reshape((H,W))
        facing_direction = pose[:3,:3].dot(np.array([[0.,0.,1.]]).T).T # 1,3
        facing_direction = facing_direction / np.linalg.norm(facing_direction)
        # depth_im is on z axis
        depth_im = (ray_direction * facing_direction).sum(-1).reshape((H,W)) * depth_on_ray

        mask_valid = ~np.isinf(depth_im.reshape(-1))

        pc = depth2pc(depth_im, calib_matrix)

        
        pose_ = sequence.first_iso.matrix.dot(np.linalg.inv(traj_data[0].reshape(4,4)).dot(pose))

        pc = (pose_[:3,:3].dot(pc[mask_valid,:].T) + pose_[:3,(3,)]).T
        color, pinds = context_map.infer(torch.from_numpy(pc).to(main_device).float())
        color = torch.clip(color, 0., 1.)

        color_im = np.zeros((H*W,3))
        color_im[mask_valid,:] = color.cpu().numpy() * 255
        color_im = color_im.reshape((H,W,3))


        # inpainting to fill the hole
        mask = (~mask_valid.reshape((H,W,1))).astype(np.uint8)
        color_im = color_im.astype(np.uint8)
        color_im = cv2.inpaint(color_im, mask,3,cv2.INPAINT_TELEA)


        #cv2.imwrite(str(render_path)+'/%d.jpg'%(id), color_im[:,:,::-1])
        cv2.imwrite(str(render_path)+'/%d.jpg'%(id), color_im[:,:,::-1])
        cv2.imwrite(str(render_path)+'/%d.png'%(id), (depth_im*6553.5).astype(np.uint16))


        










