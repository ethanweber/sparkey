"""
This executable is used to test the projection code used in our model.
"""

import tensorflow as tf
import argparse
import cv2
import numpy as np

# import the Transformer used in the model code
from models.research.keypointnet.main import Transformer
from read_tfrecords import OccnetDataLoader


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

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.handle_click)

        self.sess = tf.InteractiveSession()
        tf.train.start_queue_runners(self.sess)

    def handle_click(self, event, x, y, flags, param):
        """
        This will handle the the click events on the GUI and invoke projection actions.
        """

        if event == cv2.EVENT_LBUTTONDOWN:
            point = (x, y)
            print("clicked at: {}".format(point))

            # turn into normalized coordinate
            # add batch dimension

            # do the projections into the other view

            # back project back

            # display slider value of some sort to show results

            # draw the first point that is clicked
            cv2.circle(self.current_image, point, 2, (0, 255, 0), -1)
            cv2.imshow("image", self.current_image)

    def run(self):
        """
        Start the visualizer and open a window with one of the images.
        """
        features = self.dataloader.get_features()

        img0 = tf.image.decode_png(features["img0"], 3).eval()
        img1 = tf.image.decode_png(features["img1"], 3).eval()

        # convert to BGR for it to look right in our visualizer
        image = np.hstack([img0[...,::-1], img1[...,::-1]])
        self.current_image_original = image.copy()
        self.current_image = image.copy()

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

    dataloader = OccnetDataLoader("datasets/00001/")

    # create the transformer class
    transformer = Transformer(128, 128, "datasets/00001/")

    features = dataloader.get_features()
    
    # create the visualizer and start it
    visualizer = Visualizer(transformer, dataloader)
    visualizer.run()

    # close all open windows upon termination with the visualizer
    cv2.destroyAllWindows()