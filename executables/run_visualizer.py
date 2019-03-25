""" executable for visualizing projections based on where you click

this is primarly for debuggin and verify projections work properly.
we are assuming the use occnet data in this case as well.

# ethan: add support for keypointnet data as well

example:
python run_visualizer.py --dataset_name 999
"""

import tensorflow as tf
import argparse
import cv2
import numpy as np
import os
import argparse

# ethan: make this import better
import sys
sys.path.append("../")
from network.main import Transformer
from data.occnet_data_loader import OccnetTfrecordLoader


class Visualizer(object):
    """
    Visualizer class to make sure all the projections are done correctly.
    """

    def __init__(self, transformer, dataloader):
        # this should be the tranformer class
        self.transformer = transformer
        self.dataloader = dataloader

        self.current_image_original = None
        self.current_image = None
        self.current_depth_image = None

        # stores the current data
        self.mv0, self.mvi0, self.mv1, self.mvi1 = [None, None, None, None]

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.handle_click)

        self.sess = tf.Session()

    def get_batch_tensor_from_keypoint(self, keypoints):
        """Constructs a network that infers the orientation of an object.

        Args:
            keypoints: [num_keypoints, 2] a Python list of keypoints

        Returns:
            [1, num_keypoints, 3] a tensor in batch form
        """
        
        # turn into numpy array
        np_keypoints = np.array(keypoints)

        # put into batch form
        batch = []
        batch.append(np_keypoints)
        batch = np.array(batch)

        # turn into tensor
        tensor = tf.convert_to_tensor(batch, dtype=tf.float32)

        # normalize in the u and v dimension before returning
        return self.transformer.scale_image_coords_to_normalized_coords(tensor)

    def handle_click(self, event, x, y, flags, param):
        """
        This will handle the the click events on the GUI and invoke projection actions.
        """

        if event == cv2.EVENT_LBUTTONDOWN:
            
            point = (x, y)
            print("clicked at: {}".format(point))

            x = int(x * 640.0/128.0)
            y = int(y * 480.0/128.0)

            # only need one channel of the depth image
            z = self.current_depth_image[y, x][0] / 1000.0

            # a list of keypoints
            keypoint_list = [ [x, y, z] ]

            # get tensor and points normalized
            uvz = self.get_batch_tensor_from_keypoint(keypoint_list)
            # uvz is what our network should be predicting

            print(self.sess.run(uvz))

            # same function used in keypointnet
            def func1(x):
                return tf.transpose(tf.reshape(x, [-1, 4, 4]), [0, 2, 1])

            mv0 = func1(tf.convert_to_tensor(self.mv0, dtype=tf.float32))
            mv1 = func1(tf.convert_to_tensor(self.mv1, dtype=tf.float32))
            mvi0 = func1(tf.convert_to_tensor(self.mvi0, dtype=tf.float32))
            mvi1 = func1(tf.convert_to_tensor(self.mvi1, dtype=tf.float32))


            homogenous_world_coords = self.transformer.unproject(uvz)
            world_coords = tf.matmul(homogenous_world_coords, mvi0)
            uvz_proj = self.transformer.project(tf.matmul(world_coords, mv1))

            [newx, newy] = self.sess.run(uvz_proj[0, 0, :2])
            newx = int((newx+1.0) * 64.0)
            # keypointnet assumes flipped, so we reverse that here
            # newy = 127 - int((newy+1.0) * 64.0)
            # ethan: taking out the flip for now
            newy = 127 - int((newy+1.0) * 64.0)

            # draw the first point that is clicked and the other one
            cv2.circle(self.current_image, point, 5, (0, 255, 0), -1)
            cv2.circle(self.current_image, (newx + 128, newy), 5, (255, 255, 0), -1)
            cv2.imshow("image", self.current_image)

    def run(self):
        """
        Start the visualizer and open a window with one of the images.
        """

        # get one data point
        img0, img0_mask, img0_depth, img1, img1_mask, img1_depth, self.mv0, self.mvi0, self.mv1, self.mvi1 = self.dataloader.get_single_example_from_batch(self.sess)

        # resize the depth and show it
        left = np.vstack([img0, img0_mask])
        right = np.vstack([img1, img1_mask])
        image = np.hstack([left, right])[...,::-1]


        self.current_image_original = image.copy()
        self.current_image = image.copy()

        # support for current depth map
        # this will help decide on the right value of z for the projection
        self.current_depth_image = img0_depth.copy()

        while True:
            # display the image and wait for a keypress
            cv2.imshow("image", self.current_image)
            key = cv2.waitKey(1) & 0xFF
        
            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                self.current_image = self.current_image_original.copy()
        
            # if the 'c' key is pressed, break from the loop
            elif key == ord("c"):
                break


# TODO(ethan): make this run in a more elagant manner with argparse values
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="999", help='The dataset name located in the datasets folder.')
    args = parser.parse_args()

    dataset_name = args.dataset_name
    dataset_dir = os.path.join("../datasets", dataset_name) + "/"

    # create the dataloader and transformer classes
    dataloader = OccnetTfrecordLoader(dataset_dir, occnet_data=True)
    transformer = Transformer(128, 128, dataset_dir, occnet=True)
    
    # create the visualizer and start it
    visualizer = Visualizer(transformer, dataloader)
    visualizer.run()

    # close all open windows upon termination with the visualizer
    cv2.destroyAllWindows()