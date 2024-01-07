import os
import torch
import logging
import numpy as np
import argparse
import threading
import open3d as o3d
import network.utility as net_util

p_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(p_dir)

import ext
from .base_map import BaseMap
import numba

from time import time

# save space, commnt it if you dont want dict but grid instead
#from utils.index import Indexer



import pdb
@numba.jit
def _get_valid_idx(base_idx: np.ndarray, query_idx: np.ndarray):
    mask = np.zeros((base_idx.shape[0], ), dtype=np.bool_)
    for vi, v in enumerate(base_idx):
        if query_idx[np.searchsorted(query_idx, v)] != v:
            mask[vi] = True
    return mask



class SurfaceMap(BaseMap):
    def __init__(self, uni_model, context_map, args: argparse.Namespace,
            latent_dim: int, device: torch.device, enable_async: bool = False):
        super().__init__(uni_model, args, latent_dim, device, enable_async)

        self.context_map = context_map

        if hasattr(args, 'margin'):
            self.margin = args.margin # margin by default is .1, for replica office2 use .01
        else:
            self.margin = .1

        # sample or derivative based
        if hasattr(args, 'GPIS_mode'):
            self.GPIS_mode = args.GPIS_mode
        else:
            self.GPIS_mode = 'sample' 

        # why use this?
        self.modifying_lock = threading.Lock()

    def integrate_keyframe(self, surface_xyz: torch.Tensor, surface_normal: torch.Tensor):
        '''
            :param surface_xyz:  (N, 3) x, y, z
            :param surface_normal: (N, 3) nx, ny, nz
        '''
        assert surface_xyz.device == surface_normal.device == self.device, \
            f"Device of map {self.device} and input observation " \
            f"{surface_xyz.device, surface_normal.device} must be the same."

        self.modifying_lock.acquire()

        # -- 1. allocate new voxels --
        '''
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(surface_xyz.cpu().numpy())
        #pcd.colors = o3d.utility.Vector3dVector(cur_rgb.cpu().numpy())
        pcd.normals = o3d.utility.Vector3dVector(surface_normal.cpu().numpy())
        o3d.io.write_point_cloud('tmp.ply', pcd)

        pdb.set_trace()
        '''

        surface_xyz_zeroed = surface_xyz - self.bound_min.unsqueeze(0)
        surface_xyz_normalized = surface_xyz_zeroed / self.voxel_size
        surface_grid_id = torch.ceil(surface_xyz_normalized).long() - 1
        surface_grid_id = self._linearize_id(surface_grid_id)

        # Remove the observations where it is sparse.
        unq_mask = None
        if self.args.prune_min_vox_obs > 0:
            _, unq_inv, unq_count = torch.unique(surface_grid_id, return_counts=True, return_inverse=True)
            unq_mask = (unq_count > self.args.prune_min_vox_obs)[unq_inv]
            surface_xyz_normalized = surface_xyz_normalized[unq_mask]
            surface_grid_id = surface_grid_id[unq_mask]
            surface_normal = surface_normal[unq_mask]
        # Identify empty cells, fill the indexer.
        invalid_surface_ind = self.indexer[surface_grid_id] == -1
        if invalid_surface_ind.sum() > 0:
            invalid_flatten_id, unq_inv = torch.unique(surface_grid_id[invalid_surface_ind], return_inverse=True)
            # we would like to only expand on orthogonal planar to normal direction
            #main_direction = torch.zeros((invalid_flatten_id.shape[0],3)).to(surface_normal)
            #main_direction[unq_inv] = surface_normal[invalid_surface_ind]
            # We expand this because we want to create some dummy voxels which helps the mesh extraction.
            invalid_flatten_id = self._expand_flatten_id(invalid_flatten_id, ensure_valid=False)
            #invalid_flatten_id = self._expand_flatten_id_orthogonal(invalid_flatten_id, main_direction, ensure_valid=False)

            invalid_flatten_id = invalid_flatten_id[self.indexer[invalid_flatten_id] == -1]
            self.allocate_block(invalid_flatten_id)

        def get_pruned_surface(enabled=True, lin_pos=None):
            # Prune useless surface points for quicker gathering (set to True to enable)
            if enabled:
                encoder_voxel_pos_exp = self._expand_flatten_id(lin_pos, False)
                # encoder_voxel_pos_exp = lin_pos
                exp_indexer = torch.zeros_like(self.indexer)
                exp_indexer[encoder_voxel_pos_exp] = 1
                focus_mask = exp_indexer[surface_grid_id] == 1
                return surface_xyz_normalized[focus_mask], surface_normal[focus_mask]
            else:
                return surface_xyz_normalized, surface_normal

        # -- 2. Get all voxels whose confidence is lower than optimization threshold and encoder them --
        # Find my voxels conditions:
        #   1) Voxel confidence < Threshold
        #   2) Voxel is valid.
        #   3) Not optimized.
        #   4) There is surface points in the [-0.5 - 0.5] range of this voxel.
        map_status = torch.zeros(np.product(self.n_xyz), device=self.device, dtype=torch.short)

        encoder_voxel_pos = self.latent_vecs_pos[torch.logical_and(self.voxel_obs_count < self.args.encoder_count_th,
                                                                   self.latent_vecs_pos >= 0)]
        map_status[encoder_voxel_pos] |= self.STATUS_CONF_BIT
        # self.map_status[surface_grid_id] |= self.STATUS_SURF_BIT

        if encoder_voxel_pos.size(0) > 0:
            pruned_surface_xyz_normalized, pruned_surface_normal = get_pruned_surface(
                                    enabled=True, lin_pos=encoder_voxel_pos)

            # Gather surface samples for encoder inference
            gathered_surface_latent_inds = []
            gathered_surface_xyzn = []
            for offset in self.integration_offsets:
                _surface_grid_id = torch.ceil(pruned_surface_xyz_normalized + offset) - 1
                for dim in range(3):
                    _surface_grid_id[:, dim].clamp_(0, self.n_xyz[dim] - 1)
                surface_relative_xyz = pruned_surface_xyz_normalized - _surface_grid_id - self.relative_network_offset
                surf_gid = self._linearize_id(_surface_grid_id.long())
                surface_latent_ind = self.indexer[surf_gid]
                in_focus_obs_mask = map_status[surf_gid] >= (self.STATUS_CONF_BIT)
                gathered_surface_latent_inds.append(surface_latent_ind[in_focus_obs_mask])
                gathered_surface_xyzn.append(torch.cat(
                    [surface_relative_xyz[in_focus_obs_mask],
                    pruned_surface_normal[in_focus_obs_mask]
                    ], dim=-1))
            gathered_surface_xyzn = torch.cat(gathered_surface_xyzn)
            gathered_surface_latent_inds = torch.cat(gathered_surface_latent_inds)
            surface_blatent_mapping, pinds, pcounts = torch.unique(gathered_surface_latent_inds, return_inverse=True,
            return_counts=True)
            pcounts = pcounts.float()

            logging.info(f"{surface_blatent_mapping.size(0)} voxels will be updated by the encoder. "
                f"Points/Voxel: avg = {pcounts.mean().item()}, "
                f"min = {pcounts.min().item()}, "
                f"max = {pcounts.max().item()}")

            if self.GPIS_mode == 'sample':
                # using normal to sample, bad to noise
                new_xyz, new_y = self.model.sample(gathered_surface_xyzn[...,:3], gathered_surface_xyzn[...,3:6], margin=self.margin)
                #valid_mask = (new_xyz.max(1)[0]<1) * (new_xyz.min(1)[0]>-1)
                #xyz = new_xyz[valid_mask,:].unsqueeze(0)
                #surface_y = new_y[valid_mask,:].unsqueeze(0)

                xyz = new_xyz.unsqueeze(0)
                surface_y = new_y.unsqueeze(0)

                with torch.no_grad():
                    _, surface_F = self.model.position_encoding(xyz, half_range=False)

                pinds = torch.cat([pinds]*2,axis=0)#[valid_mask]
                #pcounts *= 2

                # extract surface code
                ##
                # scatter method [fast: .5s]
                ##
                if True: # low memory
                    step = int(1e3)
                    encoder_latent_sums = []
                    for idx in range(0, surface_blatent_mapping.shape[0], step):
                        ub = min(idx+step, surface_blatent_mapping.shape[0])
                        in_range_mask = (pinds >= idx) * (pinds < ub)
                        encoder_latent_sum = self.model.scatter_color_encoding(surface_F[:,:,in_range_mask], (surface_y.transpose(1,2))[:,:,in_range_mask], pinds[in_range_mask]-idx, s_p_2 = 1, sigma_2=1e-6).squeeze(-1).detach() * pcounts[idx:ub].view(-1,1) + self.latent_vecs[surface_blatent_mapping[idx:ub],:] * self.voxel_obs_count[surface_blatent_mapping[idx:ub]].view(-1,1)
                        encoder_latent_sums.append(encoder_latent_sum)
                    encoder_latent_sum = torch.cat(encoder_latent_sums,axis=0)

                else:

                    encoder_latent_sum = self.model.scatter_color_encoding(surface_F, surface_y.transpose(1,2), pinds, s_p_2 = 1, sigma_2=1e-6).squeeze(-1).detach() * pcounts.unsqueeze(-1) + self.latent_vecs[surface_blatent_mapping,:] * self.voxel_obs_count[surface_blatent_mapping].unsqueeze(-1)
         
            else: # Derivative based GPIS
                _, surface_F = self.model.position_encoding(gathered_surface_xyzn[...,:3], half_range=False)
                surface_J = self.model.jacobian(gathered_surface_xyzn[...,:3])
                #surface_F = torch.cat([surface_F.unsqueeze(-1),surface_J],axis=-1) # 1, L, N, 4
                #L = surface_F.shape[1]
                surface_y = gathered_surface_xyzn[...,3:6].unsqueeze(0)#torch.cat([torch.zeros((gathered_surface_xyzn.shape[0],1)).to(gathered_surface_xyzn), gathered_surface_xyzn[...,3:6]], axis=-1) # N,4
                #pinds = torch.stack([pinds]*4,axis=-1) # N,4

                #surface_F = surface_F.reshape((1,L,-1)) # 1,L,4N
                #surface_y = surface_F.reshape((1,-1,1)) # 1,4N,1
                #pinds = pinds.reshape(-1)
                if True:
                    step = int(1e3)
                    encoder_latent_sums = []
                    for idx in range(0, surface_blatent_mapping.shape[0], step):
                        ub = min(idx+step, surface_blatent_mapping.shape[0])
                        in_range_mask = (pinds >= idx) * (pinds < ub)
                        encoder_latent_sum = self.model.scatter_surface_encoding(surface_F[:,:,in_range_mask], surface_J[:,:,in_range_mask,:], surface_y[:,in_range_mask,:], pinds[in_range_mask]-idx, s_p_2 = 1).squeeze(-1).detach() * pcounts[idx:ub].unsqueeze(-1) + self.latent_vecs[surface_blatent_mapping[idx:ub],:] * self.voxel_obs_count[surface_blatent_mapping[idx:ub]].unsqueeze(-1)

                        encoder_latent_sums.append(encoder_latent_sum)
                    encoder_latent_sum = torch.cat(encoder_latent_sums,axis=0)


                else:
                    encoder_latent_sum = self.model.scatter_surface_encoding(surface_F, surface_J, surface_y, pinds, s_p_2 = 1).squeeze(-1).detach() * pcounts.unsqueeze(-1) + self.latent_vecs[surface_blatent_mapping,:] * self.voxel_obs_count[surface_blatent_mapping].unsqueeze(-1)


            self.voxel_obs_count[surface_blatent_mapping] += pcounts
            self.latent_vecs[surface_blatent_mapping] = encoder_latent_sum / self.voxel_obs_count[surface_blatent_mapping].unsqueeze(-1)

            self._mark_updated_vec_id(surface_blatent_mapping)

            torch.cuda.empty_cache()
        map_status.zero_()
        self.modifying_lock.release()

        return



    def get_fast_preview_visuals(self):
        occupied_flatten_id = torch.where(self.indexer != -1)[0]  # (B, )
        blk_verts = [self._unlinearize_id(occupied_flatten_id) * self.voxel_size + self.bound_min]
        n_block = blk_verts[0].size(0)
        blk_edges = []
        for vert_offset in [[0.0, 0.0, self.voxel_size], [0.0, self.voxel_size, 0.0],
                            [0.0, self.voxel_size, self.voxel_size], [self.voxel_size, 0.0, 0.0],
                            [self.voxel_size, 0.0, self.voxel_size], [self.voxel_size, self.voxel_size, 0.0],
                            [self.voxel_size, self.voxel_size, self.voxel_size]]:
            blk_verts.append(
                blk_verts[0] + torch.tensor(vert_offset, dtype=torch.float32, device=blk_verts[0].device).unsqueeze(0)
            )
        for vert_edge in [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]:
            blk_edges.append(np.stack([np.arange(n_block, dtype=np.int32) + vert_edge[0] * n_block,
                                       np.arange(n_block, dtype=np.int32) + vert_edge[1] * n_block], axis=1))
        blk_verts = torch.cat(blk_verts, dim=0).cpu().numpy().astype(np.float64)
        blk_wireframe = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(blk_verts),
            lines=o3d.utility.Vector2iVector(np.concatenate(blk_edges, axis=0)))
        from utils import vis_util
        return [
            blk_wireframe,
            vis_util.wireframe_bbox(self.bound_min.cpu().numpy(),
                                    self.bound_max.cpu().numpy(), color_id=4)
        ]


    def extract_mesh(self, voxel_resolution: int, max_n_triangles: int, fast: bool = True,
                     max_std: float = 2000.0, extract_async: bool = False, no_cache: bool = False,
                     interpolate: bool = True):
        """
        Extract mesh using marching cubes.
        :param voxel_resolution: int, number of sub-blocks within an LIF block.
        :param max_n_triangles: int, maximum number of triangles.
        :param fast: whether to hierarchically extract sdf for speed improvement.
        :param interpolate: whether to interpolate sdf values.
        :param extract_async: if set to True, the function will only return a mesh when
                1) There is a change in the map.
                2) The request is completed.
                otherwise, it will just return None.
        :param no_cache: ignore cached mesh and restart over.
        :return: Open3D mesh.
        """
        if self.meshing_thread is not None:
            if not self.meshing_thread.is_alive():
                self.meshing_thread = None
                self.meshing_thread_id = -1
                self.backup_vars = {}
                return self._make_mesh_from_cache()
            elif not extract_async:
                self.meshing_thread.join()
                return self._make_mesh_from_cache()
            else:
                return None

        with self.modifying_lock:
            if self.mesh_cache.updated_vec_id.size(0) == 0 and not no_cache:
                return self._make_mesh_from_cache() if not extract_async else None
            else:
                # We can start meshing, Yay!
                if no_cache:
                    updated_vec_id = torch.arange(self.n_occupied, device=self.device)
                    self.mesh_cache.clear_all()
                else:
                    updated_vec_id = self.mesh_cache.updated_vec_id
                    self.mesh_cache.clear_updated_vec()
                if extract_async:
                    for b_name in self.backup_var_names:
                        self.backup_vars[b_name] = self.cold_vars[b_name]

        def do_meshing(voxel_resolution):
            torch.cuda.synchronize()
            with torch.cuda.stream(self.meshing_stream):
                focused_flatten_id = self.latent_vecs_pos[updated_vec_id]
                occupied_flatten_id = self._expand_flatten_id(focused_flatten_id)
                
                if False:
                    # no use
                    # remove bad voxel
                    voxel_xyz = (self._unlinearize_id(occupied_flatten_id)+.5) * self.voxel_size + self.bound_min.unsqueeze(0)
                    xyz_zeroed = voxel_xyz - self.context_map.bound_min.unsqueeze(0)
                    xyz_normalized = xyz_zeroed / self.context_map.voxel_size
                    vertex = torch.ceil(xyz_normalized) - 1
                    grid_id = self.context_map._linearize_id(vertex.long())
                    _pinds = self.context_map.indexer[grid_id]
                    occupied_flatten_id = occupied_flatten_id[_pinds!=-1]


                occupied_vec_id = self.indexer[occupied_flatten_id]  # (B, )
                # Remove voxels with too low confidence.
                occupied_vec_id = occupied_vec_id[self.voxel_obs_count[occupied_vec_id] > self.args.ignore_count_th]

                vec_id_batch_mapping = torch.ones((occupied_vec_id.max().item() + 1,), device=self.device, dtype=torch.int) * -1
                vec_id_batch_mapping[occupied_vec_id] = torch.arange(0, occupied_vec_id.size(0), device=self.device,
                                                                     dtype=torch.int)
                occupied_latent_vecs = self.latent_vecs[occupied_vec_id]  # (B, 125)
                B = occupied_latent_vecs.size(0)

                # Sample more data.
                sample_a = -(voxel_resolution // 2) * (1. / voxel_resolution)
                sample_b = 1. + (voxel_resolution - 1) // 2 * (1. / voxel_resolution)

                voxel_resolution *= 2

                low_resolution = voxel_resolution // 2 if fast else voxel_resolution

                low_samples = net_util.get_samples(low_resolution, self.device, a=sample_a, b=sample_b) - \
                              self.relative_network_offset # (l**3, 3)
                low_samples = low_samples.unsqueeze(0).repeat(B, 1, 1)  # (B, l**3, 3)
                low_latents = occupied_latent_vecs.unsqueeze(1).repeat(1, low_samples.size(1), 1)  # (B, l**3, 3)
                with torch.no_grad():
                    low_sdf, low_std = net_util.forward_model(self.model,
                                                        latent_input=low_latents.view(-1, low_latents.size(-1)),
                                                        xyz_input=low_samples.view(-1, low_samples.size(-1)), max_sample=2**16)

                if fast:
                    low_sdf = low_sdf.reshape(B, 1, low_resolution, low_resolution, low_resolution)  # (B, 1, l, l, l)
                    low_std = low_std.reshape(B, 1, low_resolution, low_resolution, low_resolution)
                    high_sdf = torch.nn.functional.interpolate(low_sdf, mode='trilinear',
                                                               size=(voxel_resolution, voxel_resolution, voxel_resolution),
                                                               align_corners=True)
                    high_std = torch.nn.functional.interpolate(low_std, mode='trilinear',
                                                               size=(voxel_resolution, voxel_resolution, voxel_resolution),
                                                               align_corners=True)
                    high_sdf = high_sdf.squeeze(0).reshape(B, voxel_resolution ** 3)  # (B, H**3)
                    high_std = high_std.squeeze(0).reshape(B, voxel_resolution ** 3)
                    high_valid_lifs, high_valid_sbs = torch.where(high_sdf.abs() < 0.05) # 0.05 use higher th to avoid hole

                    if high_valid_lifs.size(0) > 0:
                        high_samples = net_util.get_samples(voxel_resolution, self.device, a=sample_a, b=sample_b) - \
                                       self.relative_network_offset  # (H**3, 3)
                        high_latents = occupied_latent_vecs[high_valid_lifs]  # (VH, 125)
                        high_samples = high_samples[high_valid_sbs]  # (VH, 3)

                        with torch.no_grad():
                            high_valid_sdf, high_valid_std = net_util.forward_model(self.model,
                                                                       latent_input=high_latents,
                                                                       xyz_input=high_samples,
                                                                       max_sample=2**16)
                        high_sdf[high_valid_lifs, high_valid_sbs] = high_valid_sdf.squeeze(-1)
                        high_std[high_valid_lifs, high_valid_sbs] = high_valid_std.squeeze(-1)

                    high_sdf = high_sdf.reshape(B, voxel_resolution, voxel_resolution, voxel_resolution)
                    high_std = high_std.reshape(B, voxel_resolution, voxel_resolution, voxel_resolution)
                else:
                    high_sdf = low_sdf.reshape(B, low_resolution, low_resolution, low_resolution)
                    high_std = low_std.reshape(B, low_resolution, low_resolution, low_resolution)

                high_sdf = -high_sdf




                # use Indexer, need to extract indexer_view
                '''
                self.indexer_view = torch.ones(np.product(self.n_xyz),device=self.device,dtype=torch.long) * -1
                keys = torch.Tensor(self.indexer.keys()).to(torch.long).to(self.device)
                self.indexer_view[keys] = self.indexer[keys]
                '''





                if interpolate:

                    vertices, vertices_flatten_id, vertices_std = system.ext.marching_cubes_interp(
                        self.indexer.view(self.n_xyz), 
                        focused_flatten_id, vec_id_batch_mapping,
                        high_sdf, high_std, max_n_triangles, self.n_xyz, max_std)  # (T, 3, 3), (T, ), (T, 3)
                else:
                    vertices, vertices_flatten_id = system.ext.marching_cubes(
                        self.indexer.view(self.n_xyz),
                        focused_flatten_id, vec_id_batch_mapping,
                        high_sdf, max_n_triangles, self.n_xyz)  # (T, 3, 3), (T, ), (T, 3)
                    vertices_std = torch.zeros((vertices.size(0), 3), dtype=torch.float32, device=vertices.device)

                vertices = vertices * self.voxel_size + self.bound_min
                vertices = vertices.cpu().numpy()
                vertices_std = vertices_std.cpu().numpy()
                # Remove relevant cached vertices and append updated/new ones.
                vertices_flatten_id = vertices_flatten_id.cpu().numpy()
                if self.mesh_cache.vertices is None:
                    self.mesh_cache.vertices = vertices
                    self.mesh_cache.vertices_flatten_id = vertices_flatten_id
                    self.mesh_cache.vertices_std = vertices_std
                else:
                    p = np.sort(np.unique(vertices_flatten_id))
                    valid_verts_idx = _get_valid_idx(self.mesh_cache.vertices_flatten_id, p)
                    self.mesh_cache.vertices = np.concatenate([self.mesh_cache.vertices[valid_verts_idx], vertices], axis=0)
                    self.mesh_cache.vertices_flatten_id = np.concatenate([
                        self.mesh_cache.vertices_flatten_id[valid_verts_idx], vertices_flatten_id
                    ], axis=0)
                    self.mesh_cache.vertices_std = np.concatenate([self.mesh_cache.vertices_std[valid_verts_idx], vertices_std], axis=0)

        if extract_async:
            self.meshing_thread = threading.Thread(target=do_meshing, args=(voxel_resolution, ))
            self.meshing_thread_id = self.meshing_thread.ident
            self.meshing_thread.daemon = True
            self.meshing_thread.start()
        else:
            do_meshing(voxel_resolution)
            return self._make_mesh_from_cache()



    def _make_mesh_from_cache(self):
        vertices = self.mesh_cache.vertices.reshape((-1, 3))
        triangles = np.arange(vertices.shape[0]).reshape((-1, 3))
        '''
        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices,
                       faces=triangles)
        trimesh.repair.fix_winding(mesh)
        #mesh.fill_holes()
        final_mesh = mesh.as_open3d

        '''
        #trimesh.smoothing.filter_laplacian(mesh)
        final_mesh = o3d.geometry.TriangleMesh()
        final_mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
        final_mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.float64))

        #final_mesh.compute_vertex_normals()
        #final_mesh = final_mesh.subdivide_midpoint(number_of_iterations=1)
        #final_mesh = final_mesh.filter_smooth_simple(number_of_iterations=1)

        '''
        print("Cluster connected triangles")
        with o3d.utility.VerbosityContextManager(
                        o3d.utility.VerbosityLevel.Debug) as cm:
                triangle_clusters, cluster_n_triangles, cluster_area = (
                                final_mesh.cluster_connected_triangles())
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        cluster_area = np.asarray(cluster_area)

        print("Show mesh with small clusters removed")
        #final_mesh = copy.deepcopy(mesh)
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < 100
        final_mesh.remove_triangles_by_mask(triangles_to_remove)
        '''






        # Assign color:
        if vertices.shape[0] > 0:
            # color to infer
            X_test = torch.from_numpy(np.asarray(final_mesh.vertices)).float().to(self.device)
            color, pinds = self.context_map.infer(X_test)
            final_mesh.vertex_colors = o3d.utility.Vector3dVector(color.detach().cpu().numpy().astype(np.float64))
            final_mesh.remove_vertices_by_index(np.where((pinds.cpu().numpy()==-1))[0])
 
            '''
            # get vid
            surface_xyz_zeroed = X_test - self.context_map.bound_min.unsqueeze(0)
            surface_xyz_normalized = surface_xyz_zeroed / self.context_map.voxel_size
            #with open('tmp2.npy', 'wb') as f:
            #    np.save(f, surface_xyz_normalized.cpu().numpy())
            #one points
            vertex = torch.ceil(surface_xyz_normalized) -1
            surface_grid_id = self.context_map._linearize_id(vertex.long())
            d_xyz = surface_xyz_normalized - vertex - 0.5

            with torch.no_grad():
                pinds = self.context_map.indexer[surface_grid_id]
                Fs = self.context_map.latent_vecs[pinds,:,:]
                # context_map v1
                #color = self.model.color_decoding(Fs.unsqueeze(0), d_xyz.unsqueeze(0))
                # context_map v2 because v2 use 8 neighbor to train feat, use half_range
                step = int(1e6)
                colors = []
                for ids in range(0,Fs.shape[0],step):
                    color = self.model.color_decoding(Fs[ids:min(Fs.shape[0], ids+step),...].unsqueeze(0), d_xyz[ids:min(Fs.shape[0], ids+step),...].unsqueeze(0)/2)
                    colors.append(color)
                color = torch.cat(colors,axis=0)
                #color = self.model.color_decoding(Fs.unsqueeze(0), d_xyz.unsqueeze(0)/2)
            '''
            

            '''
            import matplotlib.cm
            vert_color = self.mesh_cache.vertices_std.reshape((-1, )).astype(float)
            if self.extract_mesh_std_range is not None:
                vcolor_min, vcolor_max = self.extract_mesh_std_range
                vert_color = np.clip(vert_color, vcolor_min, vcolor_max)
            else:
                vcolor_min, vcolor_max = vert_color.min(), vert_color.max()
            vert_color = (vert_color - vcolor_min) / (vcolor_max - vcolor_min)
            vert_color = matplotlib.cm.jet(vert_color)[:, :3]
            final_mesh.vertex_colors = o3d.utility.Vector3dVector(vert_color)
            '''
       #o3d.io.write_triangle_mesh('tmp.ply', final_mesh)

        self.final_mesh = final_mesh
        return final_mesh

    def get_sdf(self, xyz: torch.Tensor):
        """
        Get the sdf value of the requested positions with computation graph built.
        :param xyz: (N, 3)
        :return: sdf: (M,), std (M,), valid_mask: (N,) with M elements being 1.
        """
        xyz_normalized = (xyz - self.bound_min.unsqueeze(0)) / self.voxel_size
        with torch.no_grad():
            grid_id = torch.ceil(xyz_normalized.detach()).long() - 1
            sample_latent_id = self.indexer[self._linearize_id(grid_id)]
            sample_valid_mask = sample_latent_id != -1
            # Prune validity by ignore-count.
            valid_valid_mask = self.voxel_obs_count[sample_latent_id[sample_valid_mask]] > self.args.ignore_count_th
            sample_valid_mask[sample_valid_mask.clone()] = valid_valid_mask
            valid_latent = self.latent_vecs[sample_latent_id[sample_valid_mask]]

        valid_xyz_rel = xyz_normalized[sample_valid_mask] - grid_id[sample_valid_mask] - self.relative_network_offset
        sdf, std = net_util.forward_model(self.model,
                                        latent_input=valid_latent, xyz_input=valid_xyz_rel, no_detach=True)
        return sdf.squeeze(-1), std.squeeze(-1), sample_valid_mask


