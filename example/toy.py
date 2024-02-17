import torch
import numpy as np

from example.util import get_modules, get_example_data
import pdb

device = torch.device("cuda", index=0)

# get mapper and tracker
sm, cm, tracker, config = get_modules(device)

# prepare data
colors, depths, customs, calib, poses = get_example_data(device)

for i in [0, 1]:
    # preprocess rgbd to point cloud
    frame_pose = tracker.track_camera(colors[i], depths[i], customs, calib, poses[i], scene = config.sequence_type)
    # transform data
    tracker_pc, tracker_normal, tracker_customs= tracker.last_processed_pc
    opt_depth = frame_pose @ tracker_pc
    opt_normal = frame_pose.rotation @ tracker_normal
    color_pc, color, color_normal = tracker.last_colored_pc
    color_pc = frame_pose @ color_pc
    color_normal = frame_pose.rotation @ color_normal if color_normal is not None else None

    # mapping pc
    sm.integrate_keyframe(opt_depth, opt_normal)
    cm.integrate_keyframe(color_pc, color, color_normal)

# mesh extraction
map_mesh = sm.extract_mesh(config.resolution, int(4e7), max_std=0.15, extract_async=False, interpolate=True)

import open3d as o3d
o3d.io.write_triangle_mesh('example/mesh.ply', map_mesh)
 

