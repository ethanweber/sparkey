"""
This file implements a class that can take data from the pytorch-dense-correspondence work and format for keypoint-net type of inputs.
"""

import sys
import os
import random
import numpy as np

# python path needed for dense object nets
data_loader_path = os.path.dirname(os.path.abspath(__file__))
don_dataset_path = os.path.join(data_loader_path, '..', 'pytorch-dense-correspondence-private/dense_correspondence/dataset')
sys.path.append(don_dataset_path)

from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset
from dense_correspondence.dataset.dense_correspondence_dataset_masked import ImageType
import dense_correspondence_manipulation.utils.utils as utils

# there are some unneccary prints
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class DonDataLoader(object):
    """
    Data loader class that takes from the pytorch-dense-correspondence dataset.
    """
    def __init__(self, config_filename='shoes_all.yaml'):

        with HiddenPrints():
        
            self.config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 'dataset', 'composite', config_filename)
            self.train_config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 'training', 'training.yaml')

            self.config = utils.getDictFromYamlFilename(self.config_filename)
            self.train_config = utils.getDictFromYamlFilename(self.train_config_filename)

            self.dataset = SpartanDataset(config=self.config)
            self.dataset.set_parameters_from_training_config(self.train_config)

        # holds centroid and radius for each scene
        # these are for min and max z values currently. maybe include x, y, and z in the future.
        # self.centroid_and_radius[scene_name]["centroid"] or self.centroid_and_radius[scene_name]["radius"]
        self.centroid_and_radius = {}

    def get_random_scene_from_object_id(self, object_id=None):
        # set to first object_id if not specified
        if object_id is None:
            object_id = list(self.dataset._single_object_scene_dict.keys())[0]
        # list of scenes from training set
        scenes = self.dataset._single_object_scene_dict[object_id]["train"]
        scene = scenes[random.randint(0, len(scenes)-1)]
        print("scene: {}".format(scene))
        return scene

    def get_frame_idx_pair_from_scene_name(self, scene_name):
        """Returns the indices of two frames in the scene."""
        frames = list(self.dataset.get_pose_data(scene_name).keys())
        frame_idx_a = frames[random.randint(0, len(frames)-1)]
        frame_idx_b = frames[random.randint(0, len(frames)-1)]
        print("frame_idx_a: {}, frame_idx_b: {}".format(frame_idx_a, frame_idx_b))
        return (frame_idx_a, frame_idx_b)

    def get_camera_intrinsics_matrix(self, scene_name):
        intrinsics = self.dataset.get_camera_intrinsics(scene_name)
        K = intrinsics.get_camera_matrix()
        return K

    def mask_is_contained(self, mask):
        """ return True if mask is fully contained in image. False otherwise
        
        inputs: mask as numpy array
        """
        
        y_max, x_max = mask.shape # (height, width)
        
        # check top and bottom rows
        for i in range(x_max):
            if mask[0, i] != 0.0 or mask[y_max-1, i] != 0.0:
                return False
        # check left and right cols
        for i in range(y_max):
            if mask[i, 0] != 0.0 or mask[i, x_max-1] != 0.0:
                return False
        return True

    def set_centroid_and_radius_for_scene(self, scene_name):
        """ sets the centroid and radius for scene with the min and max z value in the scene
        """

        all_frames = list(self.dataset.get_pose_data(scene_name).keys())
        global_min_depth = float("inf")
        global_max_depth = 0.0
        for frame in all_frames:
            # ethan: this might be fragile, so come back to it
            try:
                rgb_a, depth_a, mask_a, pose_a = self.dataset.get_rgbd_mask_pose(scene_name, frame)
                masked_depth = np.array(mask_a)*np.array(depth_a)
                min_depth = masked_depth[masked_depth > 0].min() / 1000.0
                max_depth = masked_depth[masked_depth > 0].max() / 1000.0
                
                global_min_depth = min(global_min_depth, min_depth)
                global_max_depth = max(global_max_depth, max_depth)
            except:
                pass
        z_min = global_min_depth
        z_max = global_max_depth

        radius = (z_max - z_min) / 2.0
        centroid = radius + z_min

        self.centroid_and_radius[scene_name] = {}
        self.centroid_and_radius[scene_name]["centroid"] = centroid
        self.centroid_and_radius[scene_name]["radius"] = radius

    def get_random_data_pair(self):
        # this will return a random data pair


        found = False
        while not found:
            # choose data from one frame
            scene_name = self.get_random_scene_from_object_id()

            # cache the values if they haven't been scene before
            if scene_name not in self.centroid_and_radius:
                self.set_centroid_and_radius_for_scene(scene_name)

            # set the cached values if this scene has not been scene before
            frame_idx_a, frame_idx_b = self.get_frame_idx_pair_from_scene_name(scene_name)

            K = self.get_camera_intrinsics_matrix(scene_name)

            rgb_a, depth_a, mask_a, pose_a = self.dataset.get_rgbd_mask_pose(scene_name, frame_idx_a)
            rgb_b, depth_b, mask_b, pose_b = self.dataset.get_rgbd_mask_pose(scene_name, frame_idx_b)

            # check that both masks are fully visible
            found = self.mask_is_contained(np.array(mask_a)) and self.mask_is_contained(np.array(mask_b))

        a_image_data = [rgb_a, depth_a, mask_a, pose_a]
        b_image_data = [rgb_b, depth_b, mask_b, pose_b]

        return K, a_image_data, b_image_data, scene_name