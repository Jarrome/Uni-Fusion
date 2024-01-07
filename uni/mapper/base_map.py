import numpy as np
import torch
import logging
import argparse
import threading
from pathlib import Path
import functools


import pdb

class BaseMap:
    def __init__(self, uni_model, args: argparse.Namespace,
            latent_dim, device: torch.device, enable_async: bool = False):


        self.model = uni_model

        self.voxel_size = args.voxel_size
        self.n_xyz = np.ceil((np.asarray(args.bound_max) - np.asarray(args.bound_min)) / args.voxel_size).astype(int).tolist()
        logging.info(f"Map size Nx = {self.n_xyz[0]}, Ny = {self.n_xyz[1]}, Nz = {self.n_xyz[2]}")

        self.args = args
        self.bound_min = torch.tensor(args.bound_min, device=device).float()
        self.bound_max = self.bound_min + self.voxel_size * torch.tensor(self.n_xyz, device=device)
        self.latent_dim = latent_dim # could be int for surface and tuple for context
        self.device = device
        self.integration_offsets = [torch.tensor(t, device=self.device, dtype=torch.float32) for t in [
            [-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5],
            [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]
        ]]
        # Directly modifiable from outside.
        self.extract_mesh_std_range = None

        self.mesh_update_affected = [torch.tensor([t], device=self.device)
                                     for t in [[-1, 0, 0], [1, 0, 0],
                                               [0, -1, 0], [0, 1, 0],
                                               [0, 0, -1], [0, 0, 1]]]
        self.relative_network_offset = torch.tensor([[0.5, 0.5, 0.5]], device=self.device, dtype=torch.float32)

        self.cold_vars = {
            "n_occupied": 0,
            "indexer": torch.ones(np.product(self.n_xyz), device=device, dtype=torch.long) * -1,
            # -- Voxel Attributes --
            # 1. Latent Vector (Geometry)
            "latent_vecs": torch.empty((1, self.latent_dim), dtype=torch.float32, device=device) if type(self.latent_dim) == int
                        else torch.empty((1, *(self.latent_dim)), dtype=torch.float32, device=device),
            # 2. Position
            "latent_vecs_pos": torch.ones((1, ), dtype=torch.long, device=device) * -1,
            # 3. Confidence on its geometry
            "voxel_obs_count": torch.zeros((1, ), dtype=torch.float32, device=device),
        }
        self.backup_var_names = ["indexer", "latent_vecs", "latent_vecs_pos", "voxel_obs_count"]

        self.backup_vars = {}
        # Allow direct visit by variable
        for p in self.cold_vars.keys():
            setattr(BaseMap, p, property(
                fget=functools.partial(BaseMap._get_var, name=p),
                fset=functools.partial(BaseMap._set_var, name=p)
            ))
        self.meshing_thread = None
        self.meshing_thread_id = -1
        self.meshing_stream = torch.cuda.Stream()
        self.mesh_cache = MeshExtractCache(self.device)
        self.latent_vecs.zero_()



    def save(self, path):
        if not isinstance(path, Path):
            path = Path(path)

        indexer_key = torch.where(self.indexer>-1)[0]
        indexer_value = self.indexer[indexer_key].copy()
        self.cold_vars['indexer_key'] = indexer_key
        self.cold_vars['indexer_value'] = indexer_value
        del self.cold_vars['indexer']

        with path.open('wb') as f:
            torch.save(self.cold_vars, f)

    def load(self, path):
        if not isinstance(path, Path):
            path = Path(path)
        with path.open('rb') as f:
            self.cold_vars = torch.load(f)

        self.cold_vars['indexer'] = torch.ones(np.product(self.n_xyz), device=device, dtype=torch.long) * -1  
        self.cold_vars['indexer'][self.cold_vars['indexer_key']] = self.cold_vars['indexer_value']


    def _get_var(self, name):
        if threading.get_ident() == self.meshing_thread_id and name in self.backup_var_names:
            return self.backup_vars[name]
        else:
            return self.cold_vars[name]

    def _set_var(self, value, name):
        if threading.get_ident() == self.meshing_thread_id and name in self.backup_var_names:
            self.backup_vars[name] = value
        else:
            self.cold_vars[name] = value

    def _inflate_latent_buffer(self, count: int):
        target_n_occupied = self.n_occupied + count
        if self.latent_vecs.size(0) < target_n_occupied:
            new_size = self.latent_vecs.size(0)
            while new_size < target_n_occupied:
                new_size *= 2
            new_vec = torch.empty((new_size, self.latent_dim), dtype=torch.float32, device=self.device) if type(self.latent_dim) == int \
                    else torch.empty((new_size, *(self.latent_dim)), dtype=torch.float32, device=self.device)
            new_vec[:self.latent_vecs.size(0)] = self.latent_vecs

            new_vec_pos = torch.ones((new_size, ), dtype=torch.long, device=self.device) * -1
            new_vec_pos[:self.latent_vecs.size(0)] = self.latent_vecs_pos
            new_voxel_conf = torch.zeros((new_size, ), dtype=torch.float32, device=self.device)
            new_voxel_conf[:self.latent_vecs.size(0)] = self.voxel_obs_count

            new_vec[self.latent_vecs.size(0):].zero_()

            self.latent_vecs = new_vec
            self.latent_vecs_pos = new_vec_pos
            self.voxel_obs_count = new_voxel_conf


        new_inds = torch.arange(self.n_occupied, target_n_occupied, device=self.device, dtype=torch.long)
        self.n_occupied = target_n_occupied
        return new_inds

    def _linearize_id(self, xyz: torch.Tensor):
        """
        :param xyz (N, 3) long id
        :return: (N, ) lineraized id to be accessed in self.indexer
        """
        return xyz[:, 2] + self.n_xyz[-1] * xyz[:, 1] + (self.n_xyz[-1] * self.n_xyz[-2]) * xyz[:, 0]

    def _unlinearize_id(self, idx: torch.Tensor):
        """
        :param idx: (N, ) linearized id for access in self.indexer
        :return: xyz (N, 3) id to be indexed in 3D
        """
        return torch.stack([idx // (self.n_xyz[1] * self.n_xyz[2]),
                            (idx // self.n_xyz[2]) % self.n_xyz[1],
                            idx % self.n_xyz[2]], dim=-1)

    def _mark_updated_vec_id(self, new_vec_id: torch.Tensor):
        """
        :param new_vec_id: (B,) updated id (indexed in latent vectors)
        """
        self.mesh_cache.updated_vec_id = torch.cat([self.mesh_cache.updated_vec_id, new_vec_id])
        self.mesh_cache.updated_vec_id = torch.unique(self.mesh_cache.updated_vec_id)

    def allocate_block(self, idx: torch.Tensor):
        """
        :param idx: (N, 3) or (N, ), if the first one, will call linearize id.
        NOTE: this will not check index overflow!
        """
        if idx.ndimension() == 2 and idx.size(1) == 3:
            idx = self._linearize_id(idx)
        new_id = self._inflate_latent_buffer(idx.size(0))
        self.latent_vecs_pos[new_id] = idx
        self.indexer[idx] = new_id

    def integrate_keyframe(self, surface_xyz: torch.Tensor, surface_normal_or_context: torch.Tensor):
        pass


    def _expand_flatten_id(self, base_flatten_id: torch.Tensor, ensure_valid: bool = True):
        expanded_flatten_id = [base_flatten_id]
        updated_pos = self._unlinearize_id(base_flatten_id)
        for affected_offset in self.mesh_update_affected:
            rs_id = updated_pos + affected_offset
            for dim in range(3):
                rs_id[:, dim].clamp_(0, self.n_xyz[dim] - 1)
            rs_id = self._linearize_id(rs_id)
            if ensure_valid:
                rs_id = rs_id[self.indexer[rs_id] != -1]
            expanded_flatten_id.append(rs_id)
        expanded_flatten_id = torch.unique(torch.cat(expanded_flatten_id))
        return expanded_flatten_id


    def _expand_flatten_id_orthogonal(self, base_flatten_id: torch.Tensor, main_direction: torch.tensor, ensure_valid: bool = True):
        expanded_flatten_id = [base_flatten_id]
        updated_pos = self._unlinearize_id(base_flatten_id)
        for affected_offset in self.mesh_update_affected:
            valid_mask = (affected_offset * main_direction).sum(1).abs() < .5 # 60 degree
            rs_id = updated_pos[valid_mask,:] + affected_offset
            for dim in range(3):
                rs_id[:, dim].clamp_(0, self.n_xyz[dim] - 1)
            rs_id = self._linearize_id(rs_id)
            if ensure_valid:
                rs_id = rs_id[self.indexer[rs_id] != -1]
            expanded_flatten_id.append(rs_id)
        expanded_flatten_id = torch.unique(torch.cat(expanded_flatten_id))
        return expanded_flatten_id

    STATUS_CONF_BIT = 1 << 0    # 1
    STATUS_SURF_BIT = 1 << 1    # 2





class MeshExtractCache:
    def __init__(self, device):
        self.vertices = None
        self.vertices_flatten_id = None
        self.vertices_std = None
        self.updated_vec_id = None
        self.device = device
        self.clear_updated_vec()

    def clear_updated_vec(self):
        self.updated_vec_id = torch.empty((0, ), device=self.device, dtype=torch.long)

    def clear_all(self):
        self.vertices = None
        self.vertices_flatten_id = None
        self.vertices_std = None
        self.updated_vec_id = None
        self.clear_updated_vec()


