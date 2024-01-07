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




class ICLNUIMDataset(object):
    '''
    path example: 'path/to/your/ICL-NUIM R-GBD Dataset/living_room_traj0_frei_png'
    '''

    cam = namedtuple('camera', 'fx fy cx cy scale')(
        481.20, 480.0, 319.5, 239.5, 5000)
    def __init__(self, path):
        path = os.path.expanduser(path)
        self.rgb = ImageReader(self.listdir(os.path.join(path, 'rgb')))
        self.depth = ImageReader(self.listdir(os.path.join(path, 'depth')))
        self.timestamps = None

    def sort(self, xs):
        return sorted(xs, key=lambda x:int(x[:-4]))

    def listdir(self, dir):
        files = [_ for _ in os.listdir(dir) if _.endswith('.png')]
        return [os.path.join(dir, _) for _ in self.sort(files)]

    def __len__(self):
        return len(self.rgb)




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

class Sfm_houseRGBDDataset(object):
    '''
    path example: 'path/to/your/TUM R-GBD Dataset/rgbd_dataset_freiburg1_xyz'
    '''
    cam = namedtuple('camera', 'fx fy cx cy scale')(
            726.21081542969,726.21081542969,359.2048034668,202.47247314453, 5000.)

    def __init__(self, path, register=True):
        path = os.path.expanduser(path)

        if not register:
            rgb_ids, rgb_timestamps = self.listdir(path, 'rgb', ext='.png')
            depth_ids, depth_timestamps = self.listdir(path, 'depth')
        else:
            rgb_imgs, rgb_timestamps = self.listdir(path, 'rgb', ext='.png')
            depth_imgs, depth_timestamps = self.listdir(path, 'depth')
            
            interval = (rgb_timestamps[1:] - rgb_timestamps[:-1]).mean() * 2/3
            matrix = np.abs(rgb_timestamps[:, np.newaxis] - depth_timestamps)
            pairs = make_pair(matrix, interval)

            rgb_ids = []
            depth_ids = []
            for i, j in pairs:
                rgb_ids.append(rgb_imgs[i])
                depth_ids.append(depth_imgs[j])

        self.rgb = ImageReader(rgb_ids, rgb_timestamps)
        self.depth = ImageReader(depth_ids, depth_timestamps)
        self.timestamps = rgb_timestamps

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
        return self.rgb[idx], self.depth[idx]

    def __len__(self):
        return len(self.rgb)



class ReplicaRGBDDataset(object):
    '''
    path example: 'path/to/your/TUM R-GBD Dataset/rgbd_dataset_freiburg1_xyz'
    '''
    cam = namedtuple('camera', 'fx fy cx cy scale')(
            600., 600., 599.5, 339.5, 6553.5)

    def __init__(self, path, start_frame: int = 0, end_frame: int = -1, first_tq: list = None, load_gt: bool = False, register=True, mesh_gt: str = None):
        path = os.path.expanduser(path)

        self.first_iso = motion_util.Isometry(q=Quaternion(array=[0.0, -1.0, 0.0, 0.0]))


        if not register:
            rgb_ids, rgb_timestamps = self.listdir(path, 'rgb', ext='.jpg')
            depth_ids, depth_timestamps = self.listdir(path, 'depth')
        else:
            rgb_imgs, rgb_timestamps = self.listdir(path, 'rgb', ext='.jpg')
            depth_imgs, depth_timestamps = self.listdir(path, 'depth')
            
            interval = (rgb_timestamps[1:] - rgb_timestamps[:-1]).mean() * 2/3
            matrix = np.abs(rgb_timestamps[:, np.newaxis] - depth_timestamps)
            pairs = make_pair(matrix, interval)

            rgb_ids = []
            depth_ids = []
            for i, j in pairs:
                rgb_ids.append(rgb_imgs[i])
                depth_ids.append(depth_imgs[j])
        rgb_ids = rgb_ids[start_frame:end_frame]
        depth_ids = depth_ids[start_frame:end_frame]
        rgb_timestamps = rgb_timestamps[start_frame:end_frame]

        self.rgb_ids = rgb_ids

        self.rgb = ImageReader(rgb_ids, rgb_timestamps, is_rgb=True)
        self.depth = ImageReader(depth_ids, depth_timestamps)
        self.timestamps = rgb_timestamps

        self.frame_id = 0

        if load_gt:
            gt_traj_path = path+'/../traj.txt'
            self.gt_trajectory = self._parse_traj_file(gt_traj_path)
            self.gt_trajectory = self.gt_trajectory[start_frame:end_frame]
            change_iso = self.first_iso.dot(self.gt_trajectory[0].inv())
            self.gt_trajectory = [change_iso.dot(t) for t in self.gt_trajectory]
            #assert len(self.gt_trajectory) == len(self.rgb)
            self.T_gt2uni = change_iso.matrix
        
        else:
            self.gt_trajectory = None
            self.T_gt2uni = self.first_iso.matrix


        if mesh_gt is None:
            print("using reconstruction mesh")
        else:
            if mesh_gt != '' and load_gt:
                self.gt_mesh = self.get_ground_truth_mesh(mesh_gt)#, gt_traj_path)
    def get_ground_truth_mesh(self, mesh_path):#, gt_traj_path):
        import trimesh
        '''
        traj_data = np.genfromtxt(gt_traj_path) 
        T0 = traj_data[0].reshape((4,4))
        change_mat = (self.first_iso.matrix.dot(np.linalg.inv(T0)))
        '''
        change_mat = self.T_gt2uni



        mesh_gt = trimesh.load(mesh_path)
        mesh_gt.apply_transform(change_mat)
        return mesh_gt.as_open3d





    def _parse_traj_file(self,traj_path):
        camera_ext = {}
        traj_data = np.genfromtxt(traj_path)
        #cano_quat = motion_util.Isometry(q=Quaternion(axis=[0.0, 0.0, 1.0], degrees=180.0))
        for id, cur_p in enumerate(traj_data):
            T = cur_p.reshape((4,4))
            #cur_q = Quaternion(imaginary=cur_p[4:7], real=cur_p[-1]).rotation_matrix
            cur_q = T[:3,:3]
            cur_t = T[:3, 3]
            #cur_q[1] = -cur_q[1]
            #cur_q[:, 1] = -cur_q[:, 1]
            #cur_t[1] = -cur_t[1]
            cur_iso = motion_util.Isometry(q=Quaternion(matrix=cur_q), t=cur_t)
            camera_ext[id] = cur_iso #cano_quat.dot(cur_iso)
        camera_ext[len(camera_ext)] = camera_ext[len(camera_ext)-1]
        return [camera_ext[t] for t in range(len(camera_ext))]





    def sort(self, xs, st = 3):
        return sorted(xs, key=lambda x:float(x[st:-4]))

    def listdir(self, path, split='rgb', ext='.png'):
        imgs, timestamps = [], []
        files = [x for x in os.listdir(os.path.join(path, split)) if x.endswith(ext)]
        st = 5
        for name in self.sort(files,st):
            imgs.append(os.path.join(path, split, name))
            timestamp = float(name[st:-len(ext)].rstrip('.'))
            timestamps.append(timestamp)

        return imgs, np.array(timestamps)

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
        return frame_data


    def __next__(self):
        frame_data = FrameData()
        if self.gt_trajectory is not None:
            frame_data.gt_pose = self.gt_trajectory[self.frame_id]
        else:
            frame_data.gt_pose = None
        frame_data.calib = FrameIntrinsic(600., 600., 599.5, 339.5, 6553.5)
        frame_data.depth =  torch.from_numpy(self.depth[self.frame_id].astype(np.float32)).cuda().float() / 6553.5
        frame_data.rgb = torch.from_numpy(self.rgb[self.frame_id]).cuda().float() / 255.
        self.frame_id += 1
        return frame_data

    def __len__(self):
        return len(self.rgb)


class NeuralSurfaceRGBDDataset(object):
    '''
    path example: 'path/to/your/TUM R-GBD Dataset/rgbd_dataset_freiburg1_xyz'
    '''
    cam = namedtuple('camera', 'fx fy cx cy scale')(
        554.2562584220408, 554.2562584220408, 320., 240., 1000)

    def __init__(self, path, register=True):
        path = os.path.expanduser(path)

        if not register:
            rgb_ids, rgb_timestamps = self.listdir(path, 'images')
            depth_ids, depth_timestamps = self.listdir(path, 'depth_filtered')
        else:
            rgb_imgs, rgb_timestamps = self.listdir(path, 'images')
            depth_imgs, depth_timestamps = self.listdir(path, 'depth_filtered')
            
            interval = (rgb_timestamps[1:] - rgb_timestamps[:-1]).mean() * 2/3
            matrix = np.abs(rgb_timestamps[:, np.newaxis] - depth_timestamps)
            pairs = make_pair(matrix, interval)

            rgb_ids = []
            depth_ids = []
            for i, j in pairs:
                rgb_ids.append(rgb_imgs[i])
                depth_ids.append(depth_imgs[j])

        self.rgb = ImageReader(rgb_ids, rgb_timestamps)
        self.depth = ImageReader(depth_ids, depth_timestamps)
        self.timestamps = rgb_timestamps

    def sort(self, xs, st = 3):
        return sorted(xs, key=lambda x:float(x[st:-4]))

    def listdir(self, path, split='rgb', ext='.png'):
        imgs, timestamps = [], []
        files = [x for x in os.listdir(os.path.join(path, split)) if x.endswith(ext)]
        st = 3 if split == "images" else 5
        for name in self.sort(files,st):
            imgs.append(os.path.join(path, split, name))
            timestamp = float(name[st:-len(ext)].rstrip('.'))
            timestamps.append(timestamp)

        return imgs, np.array(timestamps)

    def __getitem__(self, idx):
        return self.rgb[idx], self.depth[idx]

    def __len__(self):
        return len(self.rgb)

class TUMRGBDDataset(object):
    '''
    path example: 'path/to/your/TUM R-GBD Dataset/rgbd_dataset_freiburg1_xyz'
    '''
    cam = namedtuple('camera', 'fx fy cx cy scale')(
        481.20, 480.0, 319.5, 239.5, 5000)
    '''

    cam = namedtuple('camera', 'fx fy cx cy scale')(
        525.0, 525.0, 319.5, 239.5, 5000)
    '''

    def __init__(self, path, register=True):
        path = os.path.expanduser(path)

        if not register:
            rgb_ids, rgb_timestamps = self.listdir(path, 'rgb')
            depth_ids, depth_timestamps = self.listdir(path, 'depth')
        else:
            rgb_imgs, rgb_timestamps = self.listdir(path, 'rgb')
            depth_imgs, depth_timestamps = self.listdir(path, 'depth')
            
            interval = (rgb_timestamps[1:] - rgb_timestamps[:-1]).mean() * 2/3
            matrix = np.abs(rgb_timestamps[:, np.newaxis] - depth_timestamps)
            pairs = make_pair(matrix, interval)

            rgb_ids = []
            depth_ids = []
            for i, j in pairs:
                rgb_ids.append(rgb_imgs[i])
                depth_ids.append(depth_imgs[j])

        self.rgb = ImageReader(rgb_ids, rgb_timestamps)
        self.depth = ImageReader(depth_ids, depth_timestamps)
        self.timestamps = rgb_timestamps

    def sort(self, xs):
        return sorted(xs, key=lambda x:float(x[:-4]))

    def listdir(self, path, split='rgb', ext='.png'):
        imgs, timestamps = [], []
        files = [x for x in os.listdir(os.path.join(path, split)) if x.endswith(ext)]
        for name in self.sort(files):
            imgs.append(os.path.join(path, split, name))
            timestamp = float(name[:-len(ext)].rstrip('.'))
            timestamps.append(timestamp)

        return imgs, np.array(timestamps)

    def __getitem__(self, idx):
        return self.rgb[idx], self.depth[idx]

    def __len__(self):
        return len(self.rgb)
