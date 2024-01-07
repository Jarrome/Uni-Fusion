import torch
import logging
import numpy as np
import argparse

from .base_map import BaseMap

import pdb
class ContextMap(BaseMap):
    def __init__(self, uni_model, args: argparse.Namespace,
            latent_dim: int, device: torch.device, enable_async: bool = False):
        super().__init__(uni_model, args, latent_dim, device, enable_async)



    def integrate_keyframe(self, surface_xyz: torch.Tensor, surface_context: torch.Tensor, surface_normal: torch.Tensor = None):
        '''
            :param surface_xyz:  (N, 3) x, y, z
            :param surface_context: (N, c)
        '''
        assert surface_xyz.device == surface_context.device == self.device, \
            f"Device of map {self.device} and input observation " \
            f"{surface_xyz.device, surface_normal.device} must be the same."
        
        # -- 1. Allocate new voxels --

        surface_xyz_zeroed = surface_xyz - self.bound_min.unsqueeze(0)
        surface_xyz_normalized = surface_xyz_zeroed / self.voxel_size
        surface_grid_id = torch.ceil(surface_xyz_normalized).long() - 1
        surface_grid_id = self._linearize_id(surface_grid_id)


        # Identify empty cells, fill the indexer.
        invalid_surface_ind = self.indexer[surface_grid_id] == -1
        if invalid_surface_ind.sum() > 0:
            invalid_flatten_id, unq_inv = torch.unique(surface_grid_id[invalid_surface_ind],return_inverse=True)

            # We expand this because we want to create some dummy voxels which helps the mesh extraction.
            if surface_normal is not None:
                main_direction = torch.zeros((invalid_flatten_id.shape[0],3)).to(surface_normal)
                main_direction[unq_inv] = surface_normal[invalid_surface_ind]
                # for replica;
                invalid_flatten_id = self._expand_flatten_id_orthogonal(invalid_flatten_id, main_direction, ensure_valid=False)


            else:
                invalid_flatten_id = self._expand_flatten_id(invalid_flatten_id, ensure_valid=False)

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
                return surface_xyz_normalized[focus_mask], surface_context[focus_mask]
            else:
                return surface_xyz_normalized, surface_context


        map_status = torch.zeros(np.product(self.n_xyz), device=self.device, dtype=torch.short)
        encoder_voxel_pos = self.latent_vecs_pos[torch.logical_and(self.voxel_obs_count < self.args.encoder_count_th,
                                                                   self.latent_vecs_pos >= 0)]
        map_status[encoder_voxel_pos] |= self.STATUS_CONF_BIT

        # color encoding
        if encoder_voxel_pos.size(0) > 0:

            pruned_surface_xyz_normalized = surface_xyz_normalized
            pruned_surface_color = surface_context
            #pruned_surface_xyz_normalized, pruned_surface_color = get_pruned_surface(
            #                        enabled=True, lin_pos=encoder_voxel_pos)


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
                    pruned_surface_color[in_focus_obs_mask]
                    ], dim=-1))
            gathered_surface_xyzn = torch.cat(gathered_surface_xyzn)
            gathered_surface_latent_inds = torch.cat(gathered_surface_latent_inds)




            '''
            _surface_grid_id = torch.ceil(pruned_surface_xyz_normalized) - 1
            for dim in range(3):
                _surface_grid_id[:, dim].clamp_(0, self.n_xyz[dim] - 1)
            surface_relative_xyz = pruned_surface_xyz_normalized - _surface_grid_id - self.relative_network_offset
            surf_gid = self._linearize_id(_surface_grid_id.long())
            surface_latent_ind = self.indexer[surf_gid]
            in_focus_obs_mask = map_status[surf_gid] >= (self.STATUS_CONF_BIT)
            gathered_surface_latent_inds.append(surface_latent_ind[in_focus_obs_mask])
            gathered_surface_xyzn.append(torch.cat(
                [surface_relative_xyz[in_focus_obs_mask],
                 pruned_surface_color[in_focus_obs_mask]
                 ], dim=-1))

            gathered_surface_xyzn = torch.cat(gathered_surface_xyzn)
            gathered_surface_latent_inds = torch.cat(gathered_surface_latent_inds)
            '''
            surface_blatent_mapping, pinds, pcounts = torch.unique(gathered_surface_latent_inds, return_inverse=True,
                                                           return_counts=True)
            pcounts = pcounts.float()

            logging.info(f"{surface_blatent_mapping.size(0)} voxels will be updated by the encoder. "
                         f"Points/Voxel: avg = {pcounts.mean().item()}, "
                         f"min = {pcounts.min().item()}, "
                         f"max = {pcounts.max().item()}")
            with torch.no_grad():
                _, color_F = self.model.position_encoding(gathered_surface_xyzn[...,:3].unsqueeze(0), half_range=False)
            # context
            color_y = gathered_surface_xyzn[...,3:].transpose(0,1).unsqueeze(0) # 1,M,N

            with torch.no_grad():

                if False: # replica
                    step = int(1e3)
                    encoder_latent_sums = []
                    for idx in range(0, surface_blatent_mapping.shape[0], step):
                        ub = min(idx+step, surface_blatent_mapping.shape[0])
                        in_range_mask = (pinds >= idx) * (pinds < ub)
                        encoder_latent_sum = self.model.scatter_color_encoding(color_F[:,:,in_range_mask], color_y[:,:,in_range_mask], pinds[in_range_mask]-idx, s_p_2 = 1,max_node_num=300).detach() * pcounts[idx:ub].view(-1,1,1) + self.latent_vecs[surface_blatent_mapping[idx:ub],:,:] * self.voxel_obs_count[surface_blatent_mapping[idx:ub]].view(-1,1,1)
                        '''
                        except Exception as e:
                            print(e)
                            pdb.set_trace()
                        '''
                        encoder_latent_sums.append(encoder_latent_sum)
                    encoder_latent_sum = torch.cat(encoder_latent_sums,axis=0)
                else:
                    encoder_latent_sum = self.model.scatter_color_encoding(color_F, color_y, pinds, s_p_2 = 1,max_node_num=300).detach() * pcounts.view(-1,1,1) + self.latent_vecs[surface_blatent_mapping,:,:] * self.voxel_obs_count[surface_blatent_mapping].view(-1,1,1)


            self.voxel_obs_count[surface_blatent_mapping] += pcounts
            self.latent_vecs[surface_blatent_mapping] = (encoder_latent_sum / self.voxel_obs_count[surface_blatent_mapping].view(-1,1,1))

            torch.cuda.empty_cache()
        map_status.zero_()



        return




    





    def infer(self, X_test):
        # get vid
        surface_xyz_zeroed = X_test - self.bound_min.unsqueeze(0)
        surface_xyz_normalized = surface_xyz_zeroed / self.voxel_size
        #with open('tmp2.npy', 'wb') as f:
        #    np.save(f, surface_xyz_normalized.cpu().numpy())
        #one points
        vertex = torch.ceil(surface_xyz_normalized) -1
        surface_grid_id = self._linearize_id(vertex.long())
        d_xyz = surface_xyz_normalized - vertex - 0.5
        ''' eight points            
            surface_relative_xyzs, surface_grid_ids = [], []
            for offset in self.integration_offsets:
                _surface_grid_id = torch.ceil(surface_xyz_normalized + offset) - 1
                for dim in range(3):
                    _surface_grid_id[:, dim].clamp_(0, self.context_map.n_xyz[dim] - 1)
                surface_relative_xyz = surface_xyz_normalized - _surface_grid_id - self.relative_network_offset
                surface_grid_id = self.context_map._linearize_id(_surface_grid_id.long())

                surface_relative_xyzs.append(surface_relative_xyz)
                surface_grid_ids.append(surface_grid_id)
            surface_relative_xyz = torch.cat(surface_relative_xyzs)
            surface_grid_id = torch.cat(surface_grid_ids)
            d_xyz = surface_relative_xyz
        '''
        with torch.no_grad():
            pinds = self.indexer[surface_grid_id].clone()
            pcounts = self.voxel_obs_count[pinds]
            # let voxel with only one observ -1
            #pinds[pcounts.clone()<3] = -1

            Fs = self.latent_vecs[pinds,:,:]
            # context_map v1
            #color = self.model.color_decoding(Fs.unsqueeze(0), d_xyz.unsqueeze(0))
            # context_map v2 because v2 use 8 neighbor to train feat, use half_range
            step = int(1e6)
            colors = []
            for ids in range(0,Fs.shape[0],step):
                color = self.model.color_decoding(Fs[ids:min(Fs.shape[0], ids+step),...].unsqueeze(0), d_xyz[ids:min(Fs.shape[0], ids+step),...].unsqueeze(0)/2)
                colors.append(color)
            color = torch.cat(colors,axis=0)

            #pcounts = self.voxel_obs_count[surface_grid_id]
        ''' eight points
            pdb.set_trace()
            color = color.reshape((-1, 8, 3))
            mask_useful = (pinds != -1).reshape((-1,8))
            count = mask_useful.sum(-1)

            pinds = count > 6

            color = (color * mask_useful.unsqueeze(-1)).sum(1)
            color[pinds,:] /= count[pinds].unsqueeze(-1)

            pinds[count<6] = -1
        '''



        return color, pinds#, pcounts


