import os, sys
import pickle
import importlib
import open3d as o3d
import cv2
import argparse
import logging
import time
import torch
import torch.nn.functional as F
import copy
from plyfile import PlyData
import matplotlib as mpl
p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(p)


from uni.utils import exp_util, vis_util

import asyncio

from uni.encoder import utility
from uni.encoder.uni_encoder_v2 import get_uni_model 

import numpy as np
from uni.mapper.surface_map import SurfaceMap
from uni.mapper.context_map_v2 import ContextMap # 8 points
from uni.mapper.latent_map import LatentMap



import pdb

import pathlib

vis_param = argparse.Namespace()
vis_param.n_left_steps = 0
vis_param.args = None
vis_param.mesh_updated = True
# color palette for nyu40 labels



if __name__ == '__main__':

    parser = exp_util.ArgumentParserX()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    # Load in network.  (args.model is the network specification)
    #model, args_model = utility.load_model(args.training_hypers, args.using_epoch)
    args.has_ir = hasattr(args, 'ir_mapping')
    args.has_saliency = hasattr(args, 'saliency_mapping')
    args.has_style = hasattr(args, 'style_mapping')
    args.has_latent = hasattr(args, 'latent_mapping')




    args.surface_mapping = exp_util.dict_to_args(args.surface_mapping)
    args.context_mapping = exp_util.dict_to_args(args.context_mapping)
    if args.has_ir:
        args.ir_mapping = exp_util.dict_to_args(args.ir_mapping)
    if args.has_saliency:
        args.saliency_mapping = exp_util.dict_to_args(args.saliency_mapping)
    if args.has_style:
        args.style_mapping = exp_util.dict_to_args(args.style_mapping)






    #if hasattr(args, "style_mapping"):
    import uni.tracker.tracker_custom as tracker
        #args.custom_mapping = exp_util.dict_to_args(args.custom_mapping)
    #else:
        #import uni.tracker.tracker as tracker


    args.tracking = exp_util.dict_to_args(args.tracking)

    # Load in sequence.
    seq_package, seq_class = args.sequence_type.split(".")
    sequence_module = importlib.import_module("uni.dataset." + seq_package)
    sequence_module = getattr(sequence_module, seq_class)
    vis_param.sequence = sequence_module(**args.sequence_kwargs)




    
    if torch.cuda.device_count() > 1:
        main_device, aux_device = torch.device("cuda", index=0), torch.device("cuda", index=1)
    elif torch.cuda.device_count() == 1:
        main_device, aux_device = torch.device("cuda", index=0), None
    else:
        assert False, "You must have one GPU."


    # Mapping model
    uni_model = get_uni_model(main_device)
    vis_param.context_map = ContextMap(uni_model, 
            args.context_mapping, uni_model.color_code_length, device=main_device,
                                        enable_async=args.run_async)
    vis_param.surface_map = SurfaceMap(uni_model, vis_param.context_map, 
            args.surface_mapping, uni_model.surface_code_length, device=main_device,
                                        enable_async=args.run_async)
    if args.has_ir:
        vis_param.ir_map = ContextMap(uni_model, 
                args.ir_mapping, uni_model.ir_code_length, device=main_device,
                                            enable_async=args.run_async)
    if args.has_saliency:
        vis_param.saliency_map = ContextMap(uni_model, 
                args.saliency_mapping, uni_model.saliency_code_length, device=main_device,
                                            enable_async=args.run_async)
    if args.has_style:
        vis_param.style_map = ContextMap(uni_model, 
                args.style_mapping, uni_model.style_code_length, device=main_device,
                                            enable_async=args.run_async)

 
    vis_param.tracker = tracker.SDFTracker(vis_param.surface_map, args.tracking)
    vis_param.args = args



    # load
    maps = dict()
    vis_param.surface_map.load(args.outdir+'/surface.lim')
    vis_param.context_map.load(args.outdir+'/color.lim')

    if args.has_ir:
        vis_param.ir_map.load(args.outdir+'/ir.lim')
        maps['ir'] = vis_param.ir_map
    if args.has_saliency:
        vis_param.saliency_map.load(args.outdir+'/saliency.lim')
        maps['saliency'] = vis_param.saliency_map
    if args.has_style:
        vis_param.style_map.load(args.outdir+'/style.lim')
        maps['style'] = vis_param.style_map

    
    #vis_param.latent_map.load(args.outdir+'/surface.lim')

    color_mesh = vis_param.surface_map.extract_mesh(vis_param.args.resolution, int(4e7), max_std=0.15,
                                                  extract_async=False, interpolate=True, no_cache=True)
    color_mesh_transformed = copy.deepcopy(color_mesh).transform(np.linalg.inv(vis_param.sequence.T_gt2uni))
    o3d.io.write_triangle_mesh(args.outdir+'/color_recons.ply', color_mesh_transformed)


    viridis_palette = mpl.colormaps['plasma'].resampled(8)
    cividis_palette = mpl.colormaps['cividis'].resampled(8)

    X_test = torch.from_numpy(np.asarray(color_mesh.vertices)).float().to(main_device)

    if True: #hasattr(args, "style_mapping"):
        meshes, LIMs = [], []
        if args.has_ir:
            ir_mesh = o3d.geometry.TriangleMesh(color_mesh)
        saliency_mesh = o3d.geometry.TriangleMesh(color_mesh)
        style_mesh = o3d.geometry.TriangleMesh(color_mesh)

        if args.has_ir:
            meshes.append(ir_mesh)#[ir_mesh, saliency_mesh, style_mesh]
            LIMs.append(vis_param.ir_map)# = [vis_param.ir_map, vis_param.saliency_map, vis_param.style_map]
        if args.has_saliency:
            meshes.append(saliency_mesh)#, style_mesh]
            LIMs.append(vis_param.saliency_map)#, vis_param.style_map]
        if args.has_style:
            meshes.append(style_mesh)
            LIMs.append(vis_param.style_map)
        for name, mesh, LIM in zip(maps.keys(), meshes, LIMs):
            v, pinds = LIM.infer(X_test)
            if v.dim() == 1 or name == 'saliency': # ir
                v_np = v.cpu().numpy()
                v_np[v_np<0] = 0
                if name == 'ir':
                    v_np /= v_np.max()
                v_np = v_np[:,0] if name == 'saliency' else v_np
                if name == 'saliency':
                    # using platte
                    v = viridis_palette(v_np)[:,:3]
                elif name == 'ir':
                    v_eq = cv2.equalizeHist((v_np*255).astype(np.uint8)) / 255
                    v = np.repeat(v_eq, 3, 1)

            else:
                v = v.detach().cpu().numpy()

            if name == 'style':
                v = v[:,::-1]

            mesh.vertex_colors = o3d.utility.Vector3dVector(v)
            mesh.remove_vertices_by_index(np.where(pinds.cpu().numpy()==-1)[0])

            # transform from LIM coordinate to original coordinate
            mesh_transformed = mesh.transform(np.linalg.inv(vis_param.sequence.T_gt2uni))
            o3d.io.write_triangle_mesh(args.outdir+'/%s_recons.ply'%name, mesh_transformed)

            #o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    if args.has_latent:
        args.latent_mapping = exp_util.dict_to_args(args.latent_mapping)
        from external.openseg import openseg_api
        print('Loading openseg model...')
        f_im, f_tx, f_classify, lang_latent_length = openseg_api.get_api()
        print('Loaded!')
        del f_im
        torch.cuda.empty_cache()
        f_im = None
 
        lang_latent_length = (uni_model.color_code_length[0], lang_latent_length) # 20,512
        vis_param.latent_map = LatentMap(uni_model, 
                args.latent_mapping, lang_latent_length, device=main_device,
                                            enable_async=args.run_async)
        vis_param.latent_map.load(args.outdir+'/latent.lim')



       

        text_options = ['sofa','desk','sit','work','wood','eat']
        for text in text_options:
            test_t = 'other,%s'%text


            F_tx = f_tx(text)
            F_tx = torch.from_numpy(F_tx).cuda(0)
            preds = []
            step = int(1e4)
            for i in range(0,X_test.shape[0],step):
                pred = vis_param.latent_map.infer(X_test[i:min(i+step,X_test.shape[0]),:], F_tx, f_classify).detach().cpu().numpy()
                preds.append(pred)
            pred = np.concatenate(preds, axis=0).reshape(-1)
            # prob to color
            v = viridis_palette(pred)[:,:3]

            latent_mesh = o3d.geometry.TriangleMesh(color_mesh)
            latent_mesh.vertex_colors = o3d.utility.Vector3dVector(v.astype(np.float64))
            #latent_mesh.remove_vertices_by_index(np.where(pinds.cpu().numpy()==-1)[0])

            # transform from LIM coordinate to original coordinate
            latent_mesh.transform(np.linalg.inv(vis_param.sequence.T_gt2uni))
            o3d.io.write_triangle_mesh(args.outdir+'/%s_recons.ply'%('lt_'+text), latent_mesh)







    #color_mesh = color_mesh.transform(np.linalg.inv(vis_param.sequence.T_gt2uni))




    
