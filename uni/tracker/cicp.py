import numpy as np
import open3d as o3d
import pdb


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh
def execute_global_registration(source_down, target_down, source_fpfh,
                                        target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
            0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
            distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result
def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                             target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
        % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
    return result

def cicp(source, target, current_transformation=np.identity(4), scale_factor=2):
    voxel_radius = [0.04*scale_factor, 0.02*scale_factor, 0.01*scale_factor]

    max_iter = [50, 30, 14]
    #current_transformation = np.identity(4)
    #print("3. Colored point cloud registration")
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        #print([iter, radius, scale])

        #print("3-1. Downsample with a voxel size %.2f" % radius)
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        #print("3-2. Estimate normal.")
        source_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        #print("3-3. Applying colored point cloud registration")
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=iter))

        if result_icp.fitness > .9:
            return current_transformation.copy()

        '''
            deprecated
        '''
        if False: #result_icp.fitness < 0.5:
            pdb.set_trace()

            # 1. centering the point cloud
            source_ct = source.get_center() #3,1
            target_ct = target.get_center()
            source_ = source.translate(-source_ct)
            target_ = target.translate(-target_ct)

            iter = max_iter[scale] * 10
            radius = voxel_radius[scale]
            #print([iter, radius, scale])

            #print("3-1. Downsample with a voxel size %.2f" % radius)
            source_down = source_.voxel_down_sample(radius)
            target_down = target_.voxel_down_sample(radius)

            #print("3-2. Estimate normal.")
            source_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
            target_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

            #print("3-3. Applying colored point cloud registration")
            result_icp = o3d.pipelines.registration.registration_colored_icp(
                source_down, target_down, radius, current_transformation,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                relative_rmse=1e-6,
                max_iteration=iter))
            '''
            o3d.io.write_point_cloud('tmp1.pcd', source)
            o3d.io.write_point_cloud('tmp2.pcd', target)

            source_down, source_fpfh = preprocess_point_cloud(source, voxel_size=radius/2)
            target_down, target_fpfh = preprocess_point_cloud(target, voxel_size=radius/2)
            result_ransac =  execute_fast_global_registration(source_down, target_down,
                                                                source_fpfh, target_fpfh,
                                                                voxel_size=radius/2)


            print('refined fitness', result_ransac.fitness)
            current_transformation = result_ransac.transformation
            '''
            current_transformation = result_icp.transformation
            current_transformation = current_transformation.copy()
            current_transformation[:3,3] += (target_ct - current_transformation[:3,:3].dot(source_ct))
            print("reg result",result_icp.fitness, result_icp.inlier_rmse)
        else:
            current_transformation = result_icp.transformation
    return current_transformation.copy()


def poseEstimate2(pre_depth_im, pre_rgb_im, cur_depth_im, cur_rgb_im, calib, current_transformation=np.identity(4)):
    ''' from Kinect fusion: https://github.com/chengkunli96/KinectFusion/blob/main/src/kinect_fusion/fusion.py#L190
    '''
    """Colored Point Cloud Registration. Park's method"""
    fx, fy, cx, cy = calib

    depth_o3d_img = o3d.geometry.Image((pre_depth_im).astype(np.float32))
    color_o3d_img = o3d.geometry.Image((pre_rgb_im).astype(np.float32))
    pre_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_o3d_img, depth_o3d_img, 1, depth_trunc=10)

    depth_o3d_img = o3d.geometry.Image((cur_depth_im).astype(np.float32))
    color_o3d_img = o3d.geometry.Image((cur_rgb_im).astype(np.float32))
    curr_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_o3d_img, depth_o3d_img, 1, depth_trunc=10.)

    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=cur_depth_im.shape[1],
        height=cur_depth_im.shape[0],
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )

    odo_init = current_transformation
    option = o3d.pipelines.odometry.OdometryOption()

    [success_hybrid_term, trans_hybrid_term,info] = o3d.pipelines.odometry.compute_rgbd_odometry(
            curr_rgbd_image, pre_rgbd_image, pinhole_camera_intrinsic, odo_init,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)

    transform = np.array(trans_hybrid_term)
    return transform
