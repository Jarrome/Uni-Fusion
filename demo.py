import importlib
import open3d as o3d
import argparse
import logging
import time
import torch
import pathlib

import asyncio

from uni.encoder import utility
from uni.encoder.uni_encoder_v2 import get_uni_model 

import numpy as np
from uni.mapper.surface_map import SurfaceMap
from uni.mapper.context_map_v2 import ContextMap # 8 points
from uni.mapper.latent_map import LatentMap

import uni.tracker.tracker_custom as tracker


from uni.utils import exp_util, vis_util

import pdb

vis_param = argparse.Namespace()
vis_param.n_left_steps = 0
vis_param.args = None
vis_param.mesh_updated = True


def key_step(vis):
    vis_param.n_left_steps = 1
    return False


def key_continue(vis):
    vis_param.n_left_steps += 1000000000000000000000000000000000
    return False


def update_geometry(geom, name, vis):
    if not isinstance(geom, list):
        geom = [geom]

    if name in vis_param.__dict__.keys():
        for t in vis_param.__dict__[name]:
            vis.remove_geometry(t, reset_bounding_box=False)
    for t in geom:
        vis.add_geometry(t, reset_bounding_box=False)
    vis_param.__dict__[name] = geom


def refresh(vis):
    st = time.time()
    if vis:
        # This spares slots for meshing thread to emit commands.
        time.sleep(0.02)


        if not vis_param.mesh_updated and vis_param.args.run_async:
                
            map_mesh = vis_param.surface_map.extract_mesh(vis_param.args.resolution, 0, extract_async=True)

            if map_mesh is not None:
                vis_param.mesh_updated = True
                update_geometry(map_mesh, "mesh_geometry", vis)

 

    if vis_param.n_left_steps == 0:
        return False
    if vis_param.sequence.frame_id >= len(vis_param.sequence):
        # save LIMs
        print('saving LIMs...')

        save_path = pathlib.Path(vis_param.args.outdir)
        save_path.mkdir(parents=True, exist_ok=True)
        if vis_param.surface_map:
            vis_param.surface_map.save(save_path/'surface.lim')
        if vis_param.context_map:
            vis_param.context_map.save(save_path/'color.lim')
        if vis_param.ir_map:
            vis_param.ir_map.save(save_path/'ir.lim')
        if vis_param.saliency_map:
            vis_param.saliency_map.save(save_path/'saliency.lim')
        if vis_param.style_map:
            vis_param.style_map.save(save_path/'style.lim')
        if vis_param.latent_map:
            vis_param.latent_map.save(save_path/'latent.lim')





         
        torch.cuda.empty_cache()
        if vis_param.surface_map and vis_param.context_map:
            if True:#'Azure' in vis_param.args.sequence_type:
                print('Extracting trajectory...')
                from pyquaternion import Quaternion
                with open(args.outdir+'/pred_traj.txt', 'w') as f:
                    for idx, pose_iso in enumerate(vis_param.tracker.all_pd_pose):
                        pose = pose_iso.matrix#np.linalg.inv(vis_param.sequence.T_gt2uni).dot( pose_iso.matrix )
                        q = Quaternion(matrix=pose[:3,:3])
                        f.write('%s %f %f %f %f %f %f %f\n'\
                                    %(vis_param.sequence.timestamps[idx*vis_param.args.track_interval],\
                                    pose[0,3], pose[1,3], pose[2,3],\
                                    q[1], q[2], q[3], q[0]\
                                    ))
                    


            print('Extracting and saving mesh...')
            map_mesh = vis_param.surface_map.extract_mesh(vis_param.args.resolution, int(4e7), max_std=0.15,
                                                      extract_async=False, interpolate=True)
            map_mesh = map_mesh.transform(np.linalg.inv(vis_param.sequence.T_gt2uni))
            o3d.io.write_triangle_mesh(args.outdir+'/final_recons.ply', map_mesh)
     


        if vis:
            return False
        else:
            raise StopIteration

    vis_param.n_left_steps -= 1

    logging.info(f"Frame ID = {vis_param.sequence.frame_id}")
    #frame_data = next(vis_param.sequence)
    vis_param.sequence.frame_id += 1
        # Do tracking.

    if (vis_param.sequence.frame_id - 1) % vis_param.args.track_interval == 0:
        frame_data = vis_param.sequence[vis_param.sequence.frame_id-1]

        # Prune invalid depths
        frame_data.depth[torch.logical_or(frame_data.depth < vis_param.args.depth_cut_min,
                                      frame_data.depth > vis_param.args.depth_cut_max)] = np.nan

        if torch.isnan(frame_data.depth).sum() > (frame_data.depth.shape[0] * frame_data.depth.shape[1] * .9):
            return True
      
        st = time.time() 
        if frame_data.gt_pose is None:
            frame_pose = vis_param.tracker.track_camera(frame_data.rgb, frame_data.depth, frame_data.customs, frame_data.calib,
                                                    vis_param.sequence.first_iso if len(vis_param.tracker.all_pd_pose) == 0 else None,
                                                    # use replica dataset need downsample color pc
                                                    scene = vis_param.args.sequence_type)
        elif vis_param.args.slam:
            # use the gt pose at init, pose may from slam
            frame_pose = vis_param.tracker.track_camera(frame_data.rgb, frame_data.depth, frame_data.customs, frame_data.calib,
                                                                            vis_param.sequence.first_iso if len(vis_param.tracker.all_pd_pose) == 0 else None,
                                                        scene = vis_param.args.sequence_type,
                                                        init_pose = frame_data.gt_pose if len(vis_param.tracker.all_pd_pose) != 0 else None)

        else:
            # use gt pose directly
            frame_pose = vis_param.tracker.track_camera(frame_data.rgb, frame_data.depth, frame_data.customs, frame_data.calib,
                                                        frame_data.gt_pose,
                                                    scene = vis_param.args.sequence_type)


            #frame_pose = vis_param.tracker.track_camera(frame_data.rgb, frame_data.depth, frame_data.calib, frame_data.gt_pose,
        print('Tracking_time:', time.time()-st)

        if vis_param.context_map:
            color_pc, color, color_normal = vis_param.tracker.last_colored_pc 

        tracker_pc, tracker_normal, tracker_customs= vis_param.tracker.last_processed_pc
        ir, saliency, style = tracker_customs # some of them are None



        if vis:
            #pc_geometry = vis_util.pointcloud(frame_pose @ tracker_pc.cpu().numpy())
            #update_geometry(pc_geometry, "pc_geometry", vis)
            update_geometry(vis_util.frame(), "frame", vis)
            update_geometry(vis_util.trajectory([t.t for t in vis_param.tracker.all_pd_pose]), "traj_geometry", vis)
            update_geometry(vis_util.camera(frame_pose, scale=0.15, color_id=3), "camera_geometry", vis)

    torch.cuda.empty_cache()
    if (vis_param.sequence.frame_id - 1) % vis_param.args.integrate_interval == 0:
        opt_depth = frame_pose @ tracker_pc
        opt_normal = frame_pose.rotation @ tracker_normal

        opt_sample_direct = opt_depth - torch.from_numpy(frame_pose.t).to(opt_depth).unsqueeze(0)
        opt_sample_direct /= torch.sqrt((opt_sample_direct**2).sum(axis=1,keepdim=True))

        if vis_param.context_map:
            color_pc = frame_pose @ color_pc
            color_normal = frame_pose.rotation @ color_normal if color_normal is not None else None
            st = time.time() 
            vis_param.context_map.integrate_keyframe(color_pc, color, color_normal)
            print('color_time:', time.time()-st)


        if vis_param.surface_map:
            st = time.time() 
            vis_param.surface_map.integrate_keyframe(opt_depth, opt_normal)
            print('surface_time:', time.time()-st)

        if vis_param.ir_map:
            st = time.time() 
            vis_param.ir_map.integrate_keyframe(opt_depth, ir)
            print('ir_time:', time.time()-st)
        if vis_param.saliency_map:
            st = time.time()
            vis_param.saliency_map.integrate_keyframe(opt_depth, saliency)
            print('saliency_time:', time.time()-st)
        if vis_param.style_map:
            st = time.time()
            vis_param.style_map.integrate_keyframe(opt_depth, style)
            print('style_time:', time.time()-st)

        if vis_param.latent_map:
            latent_pc, latent_normal, latent = vis_param.tracker.last_latent_pc 
            latent_pc = frame_pose @ latent_pc
            st = time.time()
            vis_param.latent_map.integrate_keyframe(latent_pc, latent)
            print('latent_time:', time.time()-st)





    if (vis_param.sequence.frame_id - 1) % vis_param.args.meshing_interval == 0:
        if vis:
            fast_preview_vis = vis_param.surface_map.get_fast_preview_visuals()
            #update_geometry(fast_preview_vis[0], "block_geometry", vis)
            update_geometry((vis_util.wireframe_bbox(vis_param.surface_map.bound_min.cpu().numpy(),
                                                     vis_param.surface_map.bound_max.cpu().numpy(), color_id=4)), "bound_box", vis)
            map_mesh = vis_param.surface_map.extract_mesh(vis_param.args.resolution, int(4e6), max_std=0.15,
                                                  extract_async=vis_param.args.run_async, interpolate=True)
            vis_param.mesh_updated = map_mesh is not None
            if map_mesh is not None:
                # Note: This may be slow:
                # map_mesh.merge_close_vertices(0.01)
                # map_mesh.compute_vertex_normals()
                update_geometry(map_mesh, "mesh_geometry", vis)
        else:
            if False:
            #pass # if not vis, no need to store mesh except for the end
            
                map_mesh = vis_param.surface_map.extract_mesh(vis_param.args.resolution, int(4e6), max_std=0.15,
                                                      extract_async=vis_param.args.run_async, interpolate=True)

                vis_param.mesh_updated = map_mesh is not None
                if map_mesh is not None:
                    map_mesh = map_mesh.transform(np.linalg.inv(vis_param.sequence.T_gt2uni))
                    o3d.io.write_triangle_mesh(args.outdir+'/%d_recons.ply'%vis_param.sequence.frame_id, map_mesh)


    return True


if __name__ == '__main__':

    parser = exp_util.ArgumentParserX()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    # Load in network.  (args.model is the network specification)
    #model, args_model = utility.load_model(args.training_hypers, args.using_epoch)

    args.has_surface = hasattr(args, 'surface_mapping')
    args.has_context = hasattr(args, 'context_mapping')

    args.has_ir = hasattr(args, 'ir_mapping')
    args.has_saliency = hasattr(args, 'saliency_mapping')
    args.has_style = hasattr(args, 'style_mapping')
    args.has_latent = hasattr(args, 'latent_mapping')
   





    #args.model = args_model
    for map_name in ['context', 'surface', 'ir', 'saliency', 'style', 'latent']:
        if getattr(args, 'has_'+map_name):
            setattr(args, map_name+'_mapping', exp_util.dict_to_args(getattr(args, map_name+'_mapping'))   )
            if map_name in ['ir', 'saliency', 'style', 'latent']:
                args.sequence_kwargs['has_'+map_name] = True
        else:
            if map_name in ['ir', 'saliency', 'style', 'latent']:
                args.sequence_kwargs['has_'+map_name] = False

    '''
    args.surface_mapping = exp_util.dict_to_args(args.surface_mapping)
    args.context_mapping = exp_util.dict_to_args(args.context_mapping)
    if args.has_saliency:
        args.saliency_mapping = exp_util.dict_to_args(args.saliency_mapping)
    if args.has_style:
        args.style_mapping = exp_util.dict_to_args(args.style_mapping)
    '''



    #args.custom_mapping = exp_util.dict_to_args(args.custom_mapping)

    args.tracking = exp_util.dict_to_args(args.tracking)

    # Load in sequence.
    seq_package, seq_class = args.sequence_type.split(".")
    sequence_module = importlib.import_module("uni.dataset." + seq_package)
    sequence_module = getattr(sequence_module, seq_class)
    if args.has_style:
        args.sequence_kwargs['style_idx'] = args.style_mapping.style_idx if hasattr(args.style_mapping, 'style_idx') else 1

    # load latent model
    if hasattr(args, 'latent_mapping'):
        from thirdparts.openseg import openseg_api
        print('Loading lseg model...')
        f_im, f_tx, f_classify, lang_latent_length = openseg_api.get_api()
        print('Loaded!')
        args.sequence_kwargs['f_im'] = f_im
     
    if not hasattr(args,'slam'):
        args.slam = False
    else:
        args.sequence_kwargs['slam'] = args.slam

    vis_param.sequence = sequence_module(**args.sequence_kwargs)

    # Mapping
    if torch.cuda.device_count() > 1:
        main_device, aux_device = torch.device("cuda", index=0), torch.device("cuda", index=1)
    elif torch.cuda.device_count() == 1:
        main_device, aux_device = torch.device("cuda", index=0), None
    else:
        assert False, "You must have one GPU."


    uni_model = get_uni_model(main_device)

    for map_name in ['context', 'surface', 'ir', 'saliency', 'style', 'latent']:
        if getattr(args, 'has_'+map_name):
            if map_name == 'surface':
                vis_param.surface_map = \
                    SurfaceMap(uni_model, vis_param.context_map, 
                                args.surface_mapping, uni_model.surface_code_length, device=main_device,
                                        enable_async=args.run_async)

            elif map_name == 'latent':
                    lang_latent_length = (uni_model.color_code_length[0], lang_latent_length) # 20,512
                    vis_param.latent_map = LatentMap(uni_model,
                            args.latent_mapping, lang_latent_length, device=main_device,
                                        enable_async=args.run_async)


            else:
                map_name_ = 'color' if map_name == 'context' else map_name

                setattr(vis_param, map_name+'_map',
                            ContextMap(uni_model, 
                                getattr(args,map_name+'_mapping'), getattr(uni_model,map_name_+'_code_length'), 
                                device=main_device,
                                enable_async=args.run_async)
                        )
        else:
            setattr(vis_param, map_name+'_map', None)


                


    '''
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
    '''

    





    vis_param.tracker = tracker.SDFTracker(vis_param.surface_map, args.tracking)
    vis_param.args = args

    if args.vis:
        # Run the engine. Internal clock driven by Open3D visualizer.
        engine = o3d.visualization.VisualizerWithKeyCallback()
        engine.create_window(window_name="Implicit SLAM", width=1280, height=720, visible=True)
        engine.register_key_callback(key=ord(","), callback_func=key_step)
        engine.register_key_callback(key=ord("."), callback_func=key_continue)
        engine.get_render_option().mesh_show_back_face = True
        engine.register_animation_callback(callback_func=refresh)
        vis_ph = vis_util.wireframe_bbox([-4., -4., -4.], [4., 4., 4.])
        engine.add_geometry(vis_ph)
        engine.remove_geometry(vis_ph, reset_bounding_box=False)
        engine.run()
        engine.destroy_window()
    else:
        key_continue(None)
        try:
            while True:
                refresh(None)
        except StopIteration:
            pass
