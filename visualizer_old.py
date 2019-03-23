"""
This executable is used to test the projection code used in our model.
"""

import tensorflow as tf
import argparse
import cv2
import numpy as np

# import the Transformer used in the model code
from models.research.keypointnet.main import Transformer
from read_tfrecords import OccnetTfrecordLoader


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

        self.mv0, self.mvi0, self.mv1, self.mvi1 = [None, None, None, None]

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.handle_click)

        self.sess = tf.Session()
        # self.sess = tf.InteractiveSession()
        # tf.train.start_queue_runners(self.sess)

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
            # TODO(ethan): deal with the z value
            # z = 0.7
            point = (x, y)

            x = int(x * 640.0/128.0)
            y = int(y * 480.0/128.0)

            # only need one channel
            z = self.current_depth_image[y, x][0] / 1000.0

            print("\ndepth: {}\n".format(z))

            # a list of keypoints
            keypoint_list = [ list(point) + [z] ]
            print("clicked at: {}".format(point))
            # print(keypoint_list)

            # get tensor and points normalized
            # uvz = self.get_batch_tensor_from_keypoint(keypoint_list)


            # to this point is good
            # world_coords = self.transformer.unproject(uvz)

            # trans = tf.matmul(mvi1[0], mv0[0])
            def func1(x):
                # return tf.transpose(tf.reshape(x, [-1, 4, 4]), [0, 2, 1])
                return tf.reshape(x, [-1, 4, 4])

            mv0 = func1(tf.convert_to_tensor(self.mv0, dtype=tf.float32))
            mv1 = func1(tf.convert_to_tensor(self.mv1, dtype=tf.float32))
            mvi0 = func1(tf.convert_to_tensor(self.mvi0, dtype=tf.float32))
            mvi1 = func1(tf.convert_to_tensor(self.mvi1, dtype=tf.float32))

            # print("...........a.............")
            # print(self.sess.run(mvi1))

            # print("...........b.............")
            # print(self.sess.run(mv0))

            # T_1_0 (transfrom from frame 0 to frame)
            trans = tf.matmul(mvi1, mv0)

            # print("..............a and b ...........")
            # print(self.sess.run(trans))


            def batch_matmul(a, b):
                return tf.reshape(
                    tf.matmul(tf.reshape(a, [-1, a.shape[2].value]), b),
                    [-1, a.shape[1].value, a.shape[2].value])


            # x = 340
            # y = 135
            # z = 0.562

            uvz = tf.constant([[[x*z, y*z, 1*z]]])
            print("uvz")
            print(uvz)
            print(self.sess.run(uvz))
            world_coords = batch_matmul(uvz, tf.transpose(self.transformer.pinv_t))
            print("world_coords")
            print(world_coords)
            print(self.sess.run(world_coords))
            homogenous_world_coords = tf.concat([world_coords, tf.ones_like(uvz[:, :, 2:])], axis=2)
            print("homogenous_world_coords")
            print(homogenous_world_coords)
            print(self.sess.run(homogenous_world_coords))
            new_homogenous_world_coords = tf.matmul(trans, tf.transpose(homogenous_world_coords, [0, 2, 1]))
            new_homogenous_world_coords = tf.transpose(new_homogenous_world_coords, [0, 2, 1])
            print("new_homogenous_world_coords")
            print(new_homogenous_world_coords)
            print(self.sess.run(new_homogenous_world_coords))
            new_world_coords = new_homogenous_world_coords / new_homogenous_world_coords[:, :, 3]
            new_world_coords = new_world_coords[:, :, :3]
            print("new_world_coords")
            print(new_world_coords)
            print(self.sess.run(new_world_coords))
            new_uvz = batch_matmul(new_world_coords, tf.transpose(self.transformer.p))
            print("new_uvz")
            print(new_uvz)
            print(self.sess.run(new_uvz))
            new_z = new_uvz[:, :, 2]
            normalized_new_uvz = new_uvz / new_z
            print("normalized_new_uvz")
            print(normalized_new_uvz)
            print(self.sess.run(normalized_new_uvz))

            [newx, newy] = self.sess.run(normalized_new_uvz[0, 0, :2])
            newx = int(newx * 128.0 / 640.0)
            newy = int(newy * 128.0 / 480.0)
            # newx = int(newx)
            # newy = int(newy)


            # # world_coords = tf.matmul(trans, world_coords)

            # # world_coords[0][0] = tf.matmul(
            # #     trans[0],
            # #     world_coords[0][0]
            # # )

            # world_coords = tf.matmul(trans, tf.transpose(world_coords, [0, 2, 1]))
            # world_coords = tf.transpose(world_coords, [0, 2, 1])

            # back = self.transformer.project(world_coords)
            # final_coords = self.transformer.scale_normalized_coords_to_image_coords(back)
            # final_coords = self.sess.run(final_coords)
            # print("final result: {}".format(final_coords))

            # [newx, newy] = final_coords[0, 0, :2]
            # newx = int(newx)
            # newy = int(newy)
            

            # turn into normalized coordinate
            # add batch dimension

            # do the projections into the other view

            # back project back

            # display slider value of some sort to show results

            # draw the first point that is clicked
            cv2.circle(self.current_image, point, 5, (0, 255, 0), -1)
            # draw where the taget point is
            # with the 128 offset
            cv2.circle(self.current_image, (newx + 128, newy), 5, (255, 255, 0), -1)
            cv2.imshow("image", self.current_image)

    def run(self):
        """
        Start the visualizer and open a window with one of the images.
        """
        # features = self.dataloader.get_features()

        # img0 = tf.image.decode_png(features["img0"], 3).eval()
        # depth0 = tf.image.decode_png(features["img0_depth"], 3).eval()
        # img1 = tf.image.decode_png(features["img1"], 3).eval()

        img0, img0_mask, img0_depth, img1, img1_mask, img1_depth, self.mv0, self.mvi0, self.mv1, self.mvi1 = self.dataloader.get_single_example_from_batch(self.sess)


        print(img0.shape)
        # convert to BGR for it to look right in our visualizer
        # image = np.hstack([img0[...,::-1], img0_depth[...,::-1]])
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


# TODO(ethan): get argument parsing code to work again
# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to the image")
# args = vars(ap.parse_args())


# TODO(ethan): make this run in a more elagant manner with argparse values
if __name__ == "__main__":
    # TODO(ethan): get data from argparse

    dataloader = OccnetTfrecordLoader("datasets/00004/")

    # create the transformer class
    transformer = Transformer(128, 128, "datasets/00004/")
    
    # create the visualizer and start it
    visualizer = Visualizer(transformer, dataloader)
    visualizer.run()

    # close all open windows upon termination with the visualizer
    cv2.destroyAllWindows()