"""
This file implements a class that can take data from the pytorch-dense-correspondence work and format for keypoint-net type of inputs.
"""

import sys
import os
import random

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
    def __init__(self, config_filename='caterpillar_only.yaml'):

        with HiddenPrints():
        
            self.config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 'dataset', 'composite', config_filename)
            self.train_config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 'training', 'training.yaml')

            self.config = utils.getDictFromYamlFilename(self.config_filename)
            self.train_config = utils.getDictFromYamlFilename(self.train_config_filename)

            self.dataset = SpartanDataset(config=self.config)
            self.dataset.set_parameters_from_training_config(self.train_config)

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

    def get_random_data_pair(self):
        # this will return a random data pair

        # choose data from one frame
        scene_name = self.get_random_scene_from_object_id()
        frame_idx_a, frame_idx_b = self.get_frame_idx_pair_from_scene_name(scene_name)

        K = self.get_camera_intrinsics_matrix(scene_name)

        rgb_a, depth_a, mask_a, pose_a = self.dataset.get_rgbd_mask_pose(scene_name, frame_idx_a)
        rgb_b, depth_b, mask_b, pose_b = self.dataset.get_rgbd_mask_pose(scene_name, frame_idx_b)

        a_image_data = [rgb_a, depth_a, mask_a, pose_a]
        b_image_data = [rgb_b, depth_b, mask_b, pose_b]

        return K, a_image_data, b_image_data