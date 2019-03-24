""" executable for dense object nets projection santity check executable

This is just a sanity check to show out the data from DON can be used to project from one image into another image.

creates `run_don_projection_sanity_check_output.png`
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import random


# ethan: make this import better
import sys
sys.path.append("../")
from data.don_data_loader import DonDataLoader
import data.utils as occnet_utils

dataloader = DonDataLoader()

# get a data pair and unpack it
K, a_image_data, b_image_data = dataloader.get_random_data_pair()
rgb_a, depth_a, mask_a, pose_a = a_image_data
rgb_b, depth_b, mask_b, pose_b = b_image_data

# print the intrinsic matrix used
print("Intrinsic Matrix: \n{}".format(K))

# specify the transforms
T_w_a = pose_a
T_w_b = pose_b
T_a_b = np.matmul(np.linalg.inv(T_w_a), T_w_b)
T_b_a = np.matmul(np.linalg.inv(T_w_b), T_w_a)

# get a set of valid coordinates
valid_coords = occnet_utils.get_coords_contained_in_mask(mask_a)
u, v = random.choice(valid_coords) # choose a random, valid coord to use
z_at_u_v = np.array(depth_a)[v,u] / 1000.0
print("Z: {}".format(z_at_u_v))

camera_plane_coord = np.array([u, v, 1])*z_at_u_v
print("Camera plane coord: {}".format(camera_plane_coord))
camera_world_coord = np.matmul(np.linalg.inv(K), camera_plane_coord)
print("Camera world coord: {}".format(camera_world_coord))

# convert to homogeneous coordinates
homogeneous_camera_world_coord = np.concatenate((camera_world_coord, np.ones(1)))
# transform from a to b
world_in_b = np.matmul(T_b_a, homogeneous_camera_world_coord)

# project the world coordinate back into the image plane
# convert back to non homogenous
world_in_b /= world_in_b[3]
print("world in b: {}".format(world_in_b))
projected_into_camera_b = np.matmul(K, world_in_b[:3])
print(projected_into_camera_b / projected_into_camera_b[2])
u_new, v_new, _ = projected_into_camera_b / projected_into_camera_b[2]

a_draw = cv2.circle(np.array(rgb_a), (int(u), int(v)), 10, (255,0,0), -1)
b_draw = cv2.circle(np.array(rgb_b), (int(u_new), int(v_new)), 10, (255,0,0), -1)

image = np.hstack([a_draw, b_draw])
plt.imshow(image)
plt.savefig('run_don_projection_sanity_check_output.png')

print("\nWrote output to run_don_projection_sanity_check_output.png")