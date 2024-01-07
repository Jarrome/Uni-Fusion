import torch
import logging
import numpy as np
import argparse

# 8 points
from .context_map_v2 import ContextMap



import pdb
class LatentMap(ContextMap):
    def __init__(self, uni_model, args: argparse.Namespace,
            latent_dim: int, device: torch.device, enable_async: bool = False):
        super().__init__(uni_model, args, latent_dim, device, enable_async)

    def infer(self, X_test, F_tx, classify):
        '''
            x: N,3 

            F_tx: the text feature
        '''
        # get vid
        surface_xyz_zeroed = X_test - self.bound_min.unsqueeze(0)
        surface_xyz_normalized = surface_xyz_zeroed / self.voxel_size

        vertex = torch.ceil(surface_xyz_normalized) - 1
        surface_grid_id = self._linearize_id(vertex.long())
        d_xyz = surface_xyz_normalized - vertex - 0.5
        with torch.no_grad():
            pinds = self.indexer[surface_grid_id]
            Fs = self.latent_vecs[pinds,:,:]
            latents = self.model.color_decoding(Fs.unsqueeze(0), d_xyz.unsqueeze(0)/2)
        seg_scores = classify(latents, F_tx)
        return seg_scores






