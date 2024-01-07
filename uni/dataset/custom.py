import torch
import cv2
import numpy as np
import open3d as o3d
from PIL import Image
import tensorflow.compat.v1 as tf


from dataset.production import *
from dataset.production.latent_map import ScanNetDataset
from dataset.production.replica import ReplicaRGBDDataset
from dataset.production.bpnet_scannet import ScanNetLatentDataset
from dataset.production.azure import AzureRGBDIDataset

from tqdm import tqdm


import pdb


class CustomScanNet(ScanNetDataset):
    def __init__(self, path, start_frame: int = 0, end_frame: int = -1, first_tq: list = None, load_gt: bool = False, train=True, mesh_gt = None, style_idx=1, has_style=True, has_ir=False, has_saliency=True, has_latent=False, f_im=None):
        super().__init__(path, start_frame, end_frame, first_tq, load_gt, train, mesh_gt)
        self.has_ir = has_ir
        self.has_saliency = has_saliency
        self.has_style = has_style
        self.has_latent = has_latent
        # saliency
        from transparent_background import Remover 
        self.saliency_detector = Remover()

        # style
        from thirdparts.style_transfer.experiments import style_api
        self.style_painting = style_api.get_api(style_idx)

        # latent
        self.latent_func = f_im

        # np_str
        self.np_image_strings = []
        for rgb_id in self.inner_dataset.color_paths:
            with tf.gfile.GFile(rgb_id, 'rb') as f:
                np_image_string = np.array([f.read()])
                self.np_image_strings.append(np_image_string)



    def __getitem__(self, idx): 
        index, rgb, depth, pose = self.inner_dataset[idx]

        frame_data = FrameData()
        frame_data.calib = FrameIntrinsic(self.fx, self.fy, self.cx, self.cy, self.depth_scaling_factor)
        frame_data.depth = depth.cuda(0).float()# torch.from_numpy(self.depth[idx_id].astype(np.float32)).cuda().float()# / self.depth_scaling_factor
        frame_data.rgb = rgb.cuda(0).float() #torch.from_numpy(self.rgb[idx_id]).cuda().float() #/ 255.

        frame_data.gt_pose = self.gt_trajectory[idx] 


        frame_data.ir = None
        img = Image.fromarray((rgb.cpu().numpy()*255).astype(np.ubyte)).convert('RGB')
        frame_data.saliency = torch.from_numpy(
                self.saliency_detector.process(img,type='map').astype(np.float32)
                ).cuda(0).float() / 255. if self.has_saliency else None

        frame_data.style = torch.from_numpy(
                self.style_painting(img)).cuda(0).float() / 255. if self.has_style else None

        H,W,_ = frame_data.rgb.shape

        frame_data.latent = torch.from_numpy(self.latent_func(self.np_image_strings[idx], H, W)).cuda(0).float() if self.has_latent else None



        frame_data.customs = [frame_data.ir, frame_data.saliency, frame_data.style, frame_data.latent]
        return frame_data

class CustomAzure(AzureRGBDIDataset):
    def __init__(self, path, start_frame: int = 0, end_frame: int = -1, first_tq: list = None, load_gt: bool = False, train=True, mesh_gt = None, style_idx=1, has_style=True, has_ir=False, has_saliency=True, has_latent=False, f_im=None):
        super().__init__(path, start_frame, end_frame, first_tq, load_gt, train, mesh_gt)
        self.has_ir = has_ir
        self.has_saliency = has_saliency
        self.has_style = has_style
        self.has_latent = has_latent
        # saliency
        from transparent_background import Remover 
        self.saliency_detector = Remover()

        # style
        from thirdparts.style_transfer.experiments import style_api
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



    def __getitem__(self, idx): 
        frame_data = FrameData()
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



class CustomReplica(ReplicaRGBDDataset):
    def __init__(self, path, start_frame: int = 0, end_frame: int = -1, first_tq: list = None, load_gt: bool = False, train=True, mesh_gt = None, style_idx=1, has_style=True, has_ir=False, has_saliency=True, has_latent=False, f_im=None):
        super().__init__(path, start_frame, end_frame, first_tq, load_gt, train, mesh_gt)
        self.has_ir = has_ir
        self.has_saliency = has_saliency
        self.has_style = has_style
        self.has_latent = has_latent
        # saliency
        from transparent_background import Remover 
        self.saliency_detector = Remover()

        # style
        from thirdparts.style_transfer.experiments import style_api
        self.style_painting = style_api.get_api(style_idx)

        # latent
        self.latent_func = f_im


    def __getitem__(self, idx): 
        #return self.rgb[idx], self.depth[idx]
        frame_data = FrameData()
        if self.gt_trajectory is not None:
            frame_data.gt_pose = self.gt_trajectory[idx]
        else:
            frame_data.gt_pose = None

        frame_data.calib = FrameIntrinsic(600., 600., 599.5, 339.5, 6553.5)
        frame_data.depth =  torch.from_numpy(self.depth[idx].astype(np.float32)).cuda().float() / 6553.5
        frame_data.rgb = torch.from_numpy(self.rgb[idx]).cuda().float() / 255.

        img = Image.fromarray((frame_data.rgb.cpu().numpy()*255).astype(np.ubyte)).convert('RGB')

        frame_data.ir = None
        frame_data.saliency = torch.from_numpy(
                self.saliency_detector.process(img,type='map').astype(np.float32)
                ).cuda().float() / 255. if self.has_saliency else None

        frame_data.style = torch.from_numpy(
                self.style_painting(img)).cuda().float() / 255. if self.has_style else None

        frame_data.latent = self.latent_func(frame_data.rgb).permute(2,3,1,0).squeeze(-1) if self.has_latent else None

        frame_data.customs = [frame_data.ir, frame_data.saliency, frame_data.style, frame_data.latent]

        return frame_data


class CustomBPNetScanNet(ScanNetLatentDataset):
    def __init__(self, path, start_frame: int = 0, end_frame: int = -1, first_tq: list = None, load_gt: bool = False, train=True, mesh_gt = None, style_idx=1, has_style=True, has_ir=False, has_saliency=True, has_latent=False, f_im=None):
        super().__init__(path, f_im, start_frame, end_frame, first_tq, load_gt, train, mesh_gt)
        self.has_ir = has_ir
        self.has_saliency = has_saliency
        self.has_style = has_style
        self.has_latent = has_latent
        # saliency
        from transparent_background import Remover 
        self.saliency_detector = Remover()

        # style
        from thirdparts.style_transfer.experiments import style_api
        self.style_painting = style_api.get_api(style_idx)

        # latent
        self.latent_func = f_im


    def __getitem__(self, idx): 
        frame_data = FrameData()
        if self.gt_trajectory is not None:
            frame_data.gt_pose = self.gt_trajectory[idx]
        else:
            frame_data.gt_pose = None
        frame_data.calib = FrameIntrinsic(*self.calib)
        frame_data.depth =  torch.from_numpy(self.depth[idx].astype(np.float32)).cuda(0).float() / 1000
        frame_data.rgb = torch.from_numpy(self.rgb[idx]).cuda(0).float() / 255.

        frame_data.ir = None
        img = Image.fromarray((frame_data.rgb.cpu().numpy()*255).astype(np.ubyte)).convert('RGB')
        frame_data.saliency = torch.from_numpy(
                self.saliency_detector.process(img,type='map').astype(np.float32)
                ).cuda().float() / 255. if self.has_saliency else None

        frame_data.style = torch.from_numpy(
                self.style_painting(img)).cuda().float() / 255. if self.has_style else None

        H,W,_ = self.rgb.shape

        frame_data.latent = torch.from_numpy(self.f_im(self.np_image_strings[idx], H, W)).cuda(0).float() if self.has_latent else None



        frame_data.customs = [frame_data.ir, frame_data.saliency, frame_data.style, frame_data.latent]
        return frame_data


