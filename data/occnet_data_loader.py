""" occnet_data_loader

This is used to load data for testing. It's not actually used by the occnet network right now.

"""

import tensorflow as tf
import cv2
import numpy as np
import glob
import pickle
import os


# make a data loader that can be used by the visualizer
class OccnetTfrecordLoader(object):
    """
    Simple dataloader class to be used with reading tfrecords.
    """

    occnet_feature_set = {
        "img0": tf.FixedLenFeature([], tf.string),
        "img0_mask": tf.FixedLenFeature([16384], tf.float32), # 128 x 128
        "img0_depth": tf.FixedLenFeature([307200], tf.float32), # 128 x 128
        "img1": tf.FixedLenFeature([], tf.string),
        "img1_mask": tf.FixedLenFeature([16384], tf.float32), # 128 x 128
        "img1_depth": tf.FixedLenFeature([307200], tf.float32), # 128 x 128
        "mv0": tf.FixedLenFeature([16], tf.float32),
        "mvi0": tf.FixedLenFeature([16], tf.float32),
        "mv1": tf.FixedLenFeature([16], tf.float32),
        "mvi1": tf.FixedLenFeature([16], tf.float32),
    }

    # if using the original data
    keypointnet_feature_set = {
        "img0": tf.FixedLenFeature([], tf.string),
        "img1": tf.FixedLenFeature([], tf.string),
        "mv0": tf.FixedLenFeature([16], tf.float32),
        "mvi0": tf.FixedLenFeature([16], tf.float32),
        "mv1": tf.FixedLenFeature([16], tf.float32),
        "mvi1": tf.FixedLenFeature([16], tf.float32),
    }

    def __init__(self, dataset_dir="", occnet_data=True):
        """
            dataset_dir - full path to dataset
            occnet_data - True if using an occnet dataset, False if using a standard keypointnet dataset
        """

        if dataset_dir == "":
            raise ValueError("You must specify a dataset directory (dataset_dir)!")

        # get the tfrecord filenames
        tfrecord_path = os.path.join(dataset_dir, '*.tfrecord')
        tfrecord_filenames = glob.glob(tfrecord_path)

        def parser(serialized_example):
            # specify which feature set is being used
            if occnet_data:
                fs = tf.parse_single_example(
                    serialized_example,
                    features=self.occnet_feature_set
                )
                # before normalizing
                fs["img0_png"] = tf.image.decode_png(fs["img0"], 4)
                fs["img1_png"] = tf.image.decode_png(fs["img1"], 4)
                # normalized
                fs["img0"] = tf.div(tf.to_float(tf.image.decode_png(fs["img0"], 4)), 255)
                fs["img1"] = tf.div(tf.to_float(tf.image.decode_png(fs["img1"], 4)), 255)

                vh, vw = 128, 128
                fs["img0"].set_shape([vh, vw, 4])
                fs["img1"].set_shape([vh, vw, 4])

                # fs["lr0"] = [fs["mv0"][0]]
                # fs["lr1"] = [fs["mv1"][0]]

                fs["lr0"] = tf.convert_to_tensor([fs["mv0"][0]])
                fs["lr1"] = tf.convert_to_tensor([fs["mv1"][0]])
            else:
                fs = tf.parse_single_example(
                    serialized_example,
                    features=self.keypointnet_feature_set
                )
                fs["img0"] = tf.div(tf.to_float(tf.image.decode_png(fs["img0"], 4)), 255)
                fs["img1"] = tf.div(tf.to_float(tf.image.decode_png(fs["img1"], 4)), 255)

                vh, vw = 128, 128
                fs["img0"].set_shape([vh, vw, 4])
                fs["img1"].set_shape([vh, vw, 4])

                # fs["lr0"] = [fs["mv0"][0]]
                # fs["lr1"] = [fs["mv1"][0]]

                fs["lr0"] = tf.convert_to_tensor([fs["mv0"][0]])
                fs["lr1"] = tf.convert_to_tensor([fs["mv1"][0]])
            return fs

        # ethan: this is set for testing purposes. it's not used by keypointnet
        batch_size = 1
        dataset = tf.data.TFRecordDataset(tfrecord_filenames)
        dataset = dataset.map(parser, num_parallel_calls=4)
        dataset = dataset.shuffle(400).repeat().batch(batch_size)
        dataset = dataset.prefetch(buffer_size=256)

        self.iterator = dataset.make_one_shot_iterator().get_next()

    def get_features(self):
        # when sess runs this, features will be extracted
        return self.iterator

    def get_single_example_from_batch(self, session):
        """
        Get a single example by passing in the session.
        """

        features = self.get_features()

        img0 = features["img0"][0,:,:,:3]
        img0_mask = features["img0_mask"][0]
        img0_depth = features["img0_depth"][0]
        img1 = features["img1"][0,:,:,:3]
        img1_mask = features["img1_mask"][0]
        img1_depth = features["img1_depth"][0]
        mv0 = features["mv0"]
        mvi0 = features["mvi0"]
        mv1 = features["mv1"]
        mvi1 = features["mvi1"]

        img0, img0_mask, img0_depth, img1, img1_mask, img1_depth, mv0, mvi0, mv1, mvi1 = session.run([img0, img0_mask, img0_depth, img1, img1_mask, img1_depth, mv0, mvi0, mv1, mvi1])

        # reshape the stuff
        img0_mask = img0_mask.reshape((128, 128))
        img0_depth = img0_depth.reshape((480, 640))
        img1_mask = img1_mask.reshape((128, 128))
        img1_depth = img1_depth.reshape((480, 640))

        # get into same shape
        mask_copy = np.zeros_like(img0)
        mask_copy[:,:,0] = mask_copy[:,:,1] = mask_copy[:,:,2] = img0_mask
        img0_mask = mask_copy * 255

        mask_copy = np.zeros_like(img1)
        mask_copy[:,:,0] = mask_copy[:,:,1] = mask_copy[:,:,2] = img1_mask
        img1_mask = mask_copy * 255

        # do the same for the depths

        depth_copy = np.zeros((480, 640, 3))
        depth_copy[:,:,0] = depth_copy[:,:,1] = depth_copy[:,:,2] = img0_depth
        img0_depth = depth_copy

        depth_copy = np.zeros((480, 640, 3))
        depth_copy[:,:,0] = depth_copy[:,:,1] = depth_copy[:,:,2] = img1_depth
        img1_depth = depth_copy

        return [img0, img0_mask, img0_depth, img1, img1_mask, img1_depth, mv0, mvi0, mv1, mvi1]




