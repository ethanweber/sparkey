# some helper functions that are either being used or might be in the future

import numpy as np


# get the coordinates and distance from the projection
# source_u, source_v in pixel coordinates 640 x 480 image size
def get_u_v_d_point_after_projection(source_u, source_v, source_pose, source_depth, target_pose, K):
    """
    This returns the u, v, target_depth for the a coordinate from one frame to other.
    
    source_u, source_v are integers
    K is the camera intrinsics matrix
    """
#     print(source_depth)
    camera_plane_coord = np.array([source_u, source_v, 1])*source_depth
    camera_world_coord = np.matmul(np.linalg.inv(K), camera_plane_coord)
    
    # convert to homogeneous coordinates
    homogeneous_camera_world_coord = np.concatenate((camera_world_coord, np.ones(1)))
    # transform from source_pose to target_pose (denoted a to b here as T_b_a)
    T_b_a = np.matmul(np.linalg.inv(target_pose), source_pose)
    world_in_b = np.matmul(T_b_a, homogeneous_camera_world_coord)
    
    # project the world coordinate back into the image plane
    # convert back to non homogenous
    world_in_b /= world_in_b[3]
    projected_into_camera_b = np.matmul(K, world_in_b[:3])
    # get the depth this way just the source_depth was used
    depth = projected_into_camera_b[2]
#     print(depth)
    
    # get the coordinates and convert them to integers
    u, v, _ = projected_into_camera_b / projected_into_camera_b[2]
    u = int(u)
    v = int(v)
    
    return u, v, depth

# code that will get the coordinates in a mask
# this can then be used to take only coordinates contained within the mask
def get_coords_contained_in_mask(mask):
    coords = []
    np_mask = np.array(mask)
    h, w = np_mask.shape
    for v in range(h):
        for u in range(w):
            value = np_mask[v, u]
            if value == 1.0:
                coords.append([u,v])
            
    return coords

# TODO(ethan): only use this code if needed
# import the Transformer used in the model code
# from models.research.keypointnet.main import Transformer
# class OccnetTransformer(Transformer):
#     """
#     Here we inherrit from Transformer in the KeypointnNet code to do projection tests with visualizer.py.
#     """

#     def __init__(self, *args, **kwargs):
#         # pass all the arguments to the parrent class
#         super(OccnetTransformer, self).__init__(*args, **kwargs)

