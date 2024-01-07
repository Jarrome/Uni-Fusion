import numpy as np
import open3d as o3d
import pdb

class RayCaster:
    def __init__(self, mesh, H, W, calib):
        '''
            calib: 3x3
        '''
        # 0. init the scene
        self.scene = o3d.t.geometry.RaycastingScene()
        self.H = H
        self.W = W
        #self.calib = calib
        self.mesh = mesh

        self.mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        obj_id = self.scene.add_triangles(self.mesh_t)


        # 1. generate image's 3d point
        x = np.arange(W)
        y = np.arange(H)
        #xv, yv = np.meshgrid(x, y, indexing='ij') # WxH
        yv, xv = np.meshgrid(y, x, indexing='ij') # HxW


        img_xyz = np.ones((H*W,3))
        img_xyz[:,0] = xv.reshape(-1)
        img_xyz[:,1] = yv.reshape(-1)
        self.img_xyz = np.linalg.inv(calib).dot(img_xyz.T).T # HWx3


    def ray_cast(self, pose):
        '''
            pose: 4x4
        '''
        # 2. direction
        source_pts =  pose[:3,(3,)].T # 1,3
        direction = (pose[:3,:3]@(self.img_xyz.T)).T # HW, 3
        direction = direction / (np.linalg.norm(direction,axis=1, keepdims=True))#+1e-8)

        rays = o3d.core.Tensor(np.concatenate([np.repeat(source_pts,direction.shape[0],0),direction],axis=1), dtype=o3d.core.Dtype.Float32)

        # debug use
        '''
        ray_pcd = o3d.geometry.PointCloud()
        ray_pcd.points = o3d.utility.Vector3dVector(source_pts + direction)
    
        o3d.visualization.draw_geometries([self.mesh, ray_pcd])
        '''

        # 3. the hit distance (depth) is in ans['t_hit']
        ans = self.scene.cast_rays(rays)

        return ans, direction
        # the hit point on mesh
        back_proj_pts = source_pts + direction * ans['t_hit'].numpy()[:,np.newaxis] # HW,3
        if not return_depth: # return point cloud
            return back_proj_pts#.reshape((self.H,self.W,3))
        else:
            # also get the color
            return back_proj_pts, ans['t_hit'].numpy()
            '''
            pdb.set_trace()
            colors = np.asarray(self.mesh.vertex_colors)
            tri_vs = np.asarray(self.mesh.triangles)

            tri_id = ans['primitive_ids'].numpy()
            invalid_mask = tri_id == self.scene.INVALID_ID
            tri_id[invalid_mask] = 0

            wanted_color = colors[tri_vs[tri_id,0],:]
            wanted_color[invalid_mask,:] = 0

            return back_proj_pts, wanted_color
            '''













