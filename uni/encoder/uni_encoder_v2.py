import sys
import os

import torch
import numpy as np
from random import sample
from time import time
import math
import scipy.linalg

import torch_geometric
import functorch




import pdb

dim_k = 20

def get_uni_model(device, model_path = './uni/encoder/position_encoder.pth'):
    global dim_k
    model = Nystrom(k=dim_k, device=device, kernel_func=Matern_helper)
    model.load_approx(model_path)
    return model

def Matern_helper(X, X2=None):
    '''
        X: N,3
    '''
    if X2 is not None:
        X2 = X2.unsqueeze(0)
    return kernel_torch_Matern(X.unsqueeze(0),X2).squeeze(0)

def kernel_torch_Matern(X,X2=None,d=7,log_ell= 0.642442605255366, log_sigma=2.095394555506737, itemize_X = False):
    '''
    Matern kernel follow https://github.com/marionmari/pyGPs/blob/master/pyGPs/Core/cov.py

    X: B,N,2
    '''
    ell = np.exp(log_ell)
    sf2 = np.exp(2*log_sigma)

    #x_max = torch.tensor([[1.3991717 , 3.85962436]]).to(X)

    x_max = torch.tensor([[1 , 1, 1]]).to(X)

    X = X*4*x_max 

    X = np.sqrt(d)*X/ell
    if X2 is None:
        X2 = X
    else:
        X2 = X2*4*x_max 
        X2 = np.sqrt(d)*X2/ell





    if itemize_X:
        dist = torch.sqrt(torch.sum( (X-X)**2 ,axis=2)) # B,N
    else:
        dist = torch.sqrt(torch.sum( (X.unsqueeze(2)-X2.unsqueeze(1))**2 ,axis=3)) # B,N,M


    def func(d,t):
        if d == 1:
            return 1
        elif d == 3:
            return 1 + t
        elif d == 5:
            return 1 + t + t*t/3.
        elif d == 7:
            return 1 + t + 2.*t*t/5. + t*t*t/15.
    def mfunc(d,t):
        return func(d,t)*torch.exp(-1*t)


    return mfunc(d,dist)








class Nystrom(torch.nn.Module):
    def __init__(self, k=10, device='cuda', kernel_func=None):
        self.dim_k = k
        self.surface_code_length = k
        self.color_code_length = (k,3)
        self.ir_code_length = (k,1)
        self.saliency_code_length = (k,3)
        self.style_code_length = (k, 3)


        self.device=device

        self.q, self.p, self.eigenvalues, self.anchor_X = None, None, None, None
        self.eigenfuncs = None
        self.kernel_func = kernel_func

    def train(self, X):
        '''this train take less than 1s
            X: Nx3 
        '''
        k = self.dim_k
        K = self.kernel_func(X)
        p, q = scipy.linalg.eigh(K.data.cpu().numpy())#, subset_by_index=[K.shape[0]-k, K.shape[0]-1])
        p = torch.from_numpy(p).to(X.device).float()[range(-1, -(k+1), -1)]
        q = torch.from_numpy(q).to(X.device).float()[:, range(-1, -(k+1), -1)]
        # p, q = torch.symeig(K, eigenvectors=True)
        eigenvalues_nystrom = p / X.shape[0]
        eigenfuncs_nystrom = lambda x: self.kernel_func(x, X) @ q / p * math.sqrt(X.shape[0])

        self.q = q
        self.p = p
        self.eigenvalues = eigenvalues_nystrom
        self.eigenfuncs = eigenfuncs_nystrom
        self.anchor_X = X



    def produce_approx(self):
        X = torch.empty(256, 3).uniform_(-.5, .5).to(self.device)
        '''
        x_ = np.linspace(-.5, .5, 11)

        x, y, z = np.meshgrid(x_, x_, x_, indexing='ij')
        X = torch.from_numpy(np.stack([x,y,z],axis=-1).reshape(-1,3)).to(self.device).float()
        '''



        self.train(X)

        self.eigenvalues_11k = self.eigenvalues.unsqueeze(0).unsqueeze(0)
        self.eigenvalues_1k1 = self.eigenvalues_11k.transpose(1,2)

    def store_approx(self, path):
        state_dict = {'q':self.q,
                        'p':self.p,
                        'eigenvalues':self.eigenvalues,
                        'anchors': self.anchor_X}
        torch.save(state_dict, path)
        '''
        with open(path, 'wb') as f:
            np.save(f, self.q)
            np.save(f, self.p)
            np.save(f, self.eigenvalues)
            np.save(f, self.anchor_X)
        '''

    def load_approx(self,path):
        '''
        with open(path, 'rb') as f:
            self.q = np.load(f)
            self.p = np.load(f)
            self.eigenvalues = np.load(f)
        '''
        state_dict = torch.load(path)
        self.q = state_dict['q']
        self.p = state_dict['p']
        self.eigenvalues = state_dict['eigenvalues']
        self.anchor_X = state_dict['anchors']

        self.eigenvalues_11k = self.eigenvalues.unsqueeze(0).unsqueeze(0)
        self.eigenvalues_1k1 = self.eigenvalues_11k.transpose(1,2)
        self.eigenfuncs = lambda x: self.kernel_func(x, self.anchor_X) @ self.q / self.p * math.sqrt(self.anchor_X.shape[0])

    def position_encoding(self, points, half_range = True):
        """ points -> features
            [B, N, 3] -> [B, K, N]
            or N,3
        """
        if points.ndim==2:
            points = points.unsqueeze(0)
        B,N,_ = points.shape
        if half_range: # train in range [-.5,.5]
            x = points
        else: # during testing in range [-1,1]
            x = points /2

        x = x.view(-1,3)
        v = self.eigenfuncs(x) # BN,k
        v = v.reshape(B,N,-1).transpose(1,2) # B,k,N
        return None, v



    def color_decoding(self, latent, x):
        '''
            latent: 1,N,M (surface latent) or 1,N,M,c ( semantic latent)
            x:      1,N,3
            
            mu:     N
        '''
        step = int(5e4)
        mus = []
        for idx in range(0, int(x.shape[1]), step):
            _, F_p = self.position_encoding(x[:,idx:min(x.shape[1],idx+step),:]) # _, 1,M,N

            if latent.dim() == 3: # surface latent
                mu = (F_p.transpose(1,2) * latent[:,idx:min(x.shape[1],idx+step),:]).sum(-1) # 1,N 
            else:
                mu = (F_p.transpose(1,2).unsqueeze(-1) * latent[:,idx:min(x.shape[1],idx+step),:,:]).sum(-2) # 1,N,c
            mus.append(mu)
        mu = torch.cat(mus,axis=1)
        return mu.squeeze()


    # for surface, all in range [-1,1], but our model is in [-.5,.5], should need x/2

    def pack_func(self, x):
        '''
            x: 3,
        '''
        x_ = x.unsqueeze(0).unsqueeze(0)
        _,F = self.position_encoding(x_, half_range=False)
        return F.squeeze(0)

    def jacobian(self, points):
        '''
            points: N,3

            J: 1,L,N,3
        '''
        N = points.shape[0]
        #J = functorch.vmap(functorch.jacrev(self.pack_func))(points) # N, L, 1, 3
        J = functorch.vmap(functorch.jacfwd(self.pack_func))(points) # N, L, 1, 3

        J = J.transpose(0,2)#.reshape((1,-1, N, 3)) #1,L,N, 3
        return J

    def sample(self, points, normals=None, center=None, margin=.05):
        '''
            points: N,3
            normals: N,3

            center: 3

            return:
                new_points: N,3
                new_y: N,1
        '''

        N = points.shape[0]
        if center is None:
            assert normals is not None, "at least give center or normal"
            # using normal, which is bad if has noise
            new_points = torch.cat([points+normals*margin, points-normals*margin],axis=0)

        else:
            # using the line between center and point
            vec = points-center.unsqueeze(0)
            vec = vec / torch.sqrt((vec**2).sum(1,keepdim=True))
            new_points = torch.cat([points+vec*margin, points-vec*margin, points],axis=0)
        '''
        new_y = torch.zeros((N*3,1)).to(points)
        new_y[:N,:] = -margin
        new_y[N:2*N,:] = margin
        '''
        new_y = torch.ones((N*2,1)).to(points) * margin
        new_y[:N,:] *= -1


        return new_points, new_y


    def surface_decoding(self, chunk):
        '''
            N,dim_k+3
        '''
        chunk = chunk.unsqueeze(0)

        mu = self.color_decoding(chunk[:,:,:self.dim_k],chunk[:,:,self.dim_k:]/2)
        std = torch.zeros_like(mu).to(mu)+.1 #torch.abs(mu)/100#torch.zeros(mu.shape).to(mu)+.01
        '''
        length = ((chunk[:,:,self.dim_k:]/2)**2).sum(-1)
        overlap_region_mask = length > 0.25
        std[overlap_region_mask[0,:]] = .2
        '''

        return mu, std

    # batching
    def scatter_surface_encoding(self, F, J, y, pinds, s_p_2=1, sigma_2=1e-6):
        '''
            F: 1, M, (N) 
            J: 1, M, N, 3
            y: 1, N, 3 # dx, dy, dz 


            pinds: must be ordered, so here we will reorder it

            res: B, M, 1
        '''

        # 0. make pinds ordered
        sorted_pinds, sort_idx = torch.sort(pinds)
        F = F[:,:,sort_idx]
        J = J[:,:,sort_idx,:]
        y = y[:,sort_idx,:]

        
        _,M, N = F.shape
        F_T_b, mask_F = torch_geometric.utils.to_dense_batch(F[0,:,:].transpose(0,1), sorted_pinds) # B, n, M     (n is max(cluster sizes))
        J_T_b, mask_J = torch_geometric.utils.to_dense_batch(J[0,:,:,:].transpose(0,1), sorted_pinds) # B, n, M, 3
        y_b, mask_y = torch_geometric.utils.to_dense_batch(y[0,:,:], sorted_pinds) # B, n, 3



        B, n, M = F_T_b.shape

        if True: # same as else
            F_cat = torch.concat([F_T_b.unsqueeze(-1), J_T_b],axis=-1) # B,n,M,4
            F_cat = F_cat.transpose(1,2).reshape((B,M,-1)) # B,M,4n
            y_cat = torch.concat([torch.zeros(B,n,1).to(y),y_b],axis=-1) # B,n,4
            y_cat = y_cat.reshape((B,1,-1))# B,1,4n

            mask_J = mask_J.unsqueeze(2).expand(B,n,3) # B,n,3
            mask_cat = torch.concat([mask_F.unsqueeze(-1),mask_J],axis=-1) # B,n,4
            mask_cat = mask_cat.reshape(B,-1) # B,4n
        else:
            J_b = J_T_b.transpose(1,2).reshape((B,M,-1)) # B, M, 3n
            y_b = y_b.reshape((B,1,-1)) # B, 1, 3n

            F_cat = torch.concat([F_T_b.transpose(1,2),J_b],axis=2) # B,M,4n
            y_cat = torch.concat([torch.zeros(B,1,n).to(y_b),y_b],axis=2)

            # mask is B,n size
            mask_J = mask_J.unsqueeze(2).expand(B,n,3).reshape(B,-1)
            mask_cat = torch.concat([mask_F, mask_J],axis=1) # B, 4n

        return self.scatter_encoding(F_cat, y_cat, mask_cat, sigma_2=1e-5)

    def scatter_color_encoding(self, F, y, pinds, s_p_2=1, sigma_2=1e-6, max_node_num=None):
        '''
            F: 1, M, (N) 
            y: 1, 3, N # rgb
            pinds: must be ordered, so here we will reorder it

            res: B, M, 3
        '''
        # 0. make pinds ordered
        sorted_pinds, sort_idx = torch.sort(pinds)
        F = F[:,:,sort_idx]
        y = y[:,:,sort_idx]

        _,M, N = F.shape
        F_T_b, mask_F = torch_geometric.utils.to_dense_batch(F[0,:,:].transpose(0,1), sorted_pinds) # B, n, M     (n is max(cluster sizes))
        y_b, _ = torch_geometric.utils.to_dense_batch(y[0,:,:].transpose(0,1), sorted_pinds) # B, n, 3
        if max_node_num is not None:
            F_T_b = F_T_b[:,:max_node_num,:]
            y_b = y_b[:,:max_node_num,:]
            mask_F = mask_F[:,:max_node_num]

        #print(F_T_b.shape)
        feats = []
        step = int(5e4)
        for idx in range(0,F_T_b.shape[0],step):
            feat = self.scatter_encoding(F_T_b[idx:min(F_T_b.shape[0],idx+step),:,:].transpose(1,2), y_b[idx:min(F_T_b.shape[0],idx+step),:,:].transpose(1,2), mask_F[idx:min(F_T_b.shape[0],idx+step),:], sigma_2=1e-2)
            feats.append(feat)
        feat = torch.cat(feats,axis=0)
        #feat = self.scatter_encoding(F_T_b.transpose(1,2), y_b.transpose(1,2), mask_F, sigma_2=1e-2)

        del F, y, F_T_b, y_b, mask_F
        torch.cuda.empty_cache()

        return feat
    def scatter_encoding(self, F, y, mask, sigma_2=1e-6):
        # B, M, N
        # B, c, N
        # B, N
        B,M,N = F.shape
        y_T = y.transpose(1,2)
        F_T = F.transpose(1,2)
        st = time()
        I_N = torch.eye(N).unsqueeze(0).to(F)
        K = (F_T*self.eigenvalues_11k).bmm(F)  # B,N,N
        # set unmasked diagonal as 1 
        masked_diag = I_N.repeat(B,1,1) # B,N,N
        masked_diag[mask,:] *= sigma_2
        K += masked_diag
        u = torch.cholesky(K)
        del F_T,y,I_N,K,masked_diag,mask
        K_inv_y = torch.cholesky_solve(y_T, u)
        #K_inv_y = torch.linalg.inv(K).bmm(y_T)
        latent = (F*self.eigenvalues_1k1).bmm(K_inv_y)
        del F, K_inv_y, y_T, u
        torch.cuda.empty_cache()
        return latent



    def NIMTRE_surface_encoding(self, points, pinds, max_node_num=None):
        # points: N, 6
        if max_node_num is not None:
            # each cluster only has at most max_node_num points
            # shuffle 
            shuffle_idx = torch.randperm(points.shape[0])
            points = points[shuffle_idx,:]
            pinds = pinds[shuffle_idx]


        p, n = points[:,:3], points[:,3:]
        new_xyz, new_y = self.sample(p, n, margin=.2)
        _, surface_F = self(new_xyz.unsqueeze(0), half_range=False)
        pinds_2 = torch.concat([pinds]*2,axis=0)
        feat = self.scatter_color_encoding(surface_F, new_y.unsqueeze(0).transpose(1,2), pinds_2, s_p_2 = 1, sigma_2=1e-6, max_node_num=max_node_num).squeeze(-1)
        del points, p, n, new_xyz, new_y, surface_F, pinds_2
        torch.cuda.empty_cache()
        return feat

        












def main():
    # model
    global dim_k
    model_pth = './uni/encoder/position_encoder.pth'
    model = Nystrom(k=dim_k, device='cuda', kernel_func=Matern_helper)
    if os.path.exists(model_pth):
        model.load_approx(model_pth)
    else:
        model.produce_approx()
        model.store_approx(model_pth)


    X_test = torch.empty(2000, 3).uniform_(-.5, .5).cuda()
    K_true = Matern_helper(X_test)
    _, v = model.position_encoding(X_test.unsqueeze(0)) # B, k, N
    K_pred = (v.transpose(1,2) * model.eigenvalues_11k).bmm(v)

    error = torch.abs(K_true-K_pred).mean()
    print("mean error:",error.item())





if __name__ == '__main__':
    main()
