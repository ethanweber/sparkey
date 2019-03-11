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

    feature_set = {
        "img0": tf.FixedLenFeature([], tf.string),
        "img0_mask": tf.FixedLenFeature([16384], tf.float32), # 128 x 128
        "img0_depth": tf.FixedLenFeature([16384], tf.float32), # 128 x 128
        "img1": tf.FixedLenFeature([], tf.string),
        "img1_mask": tf.FixedLenFeature([16384], tf.float32), # 128 x 128
        "img1_depth": tf.FixedLenFeature([16384], tf.float32), # 128 x 128
        "mv0": tf.FixedLenFeature([16], tf.float32),
        "mvi0": tf.FixedLenFeature([16], tf.float32),
        "mv1": tf.FixedLenFeature([16], tf.float32),
        "mvi1": tf.FixedLenFeature([16], tf.float32),
    }

    # if using the original data
    keypointnet_features = {
        "img0": tf.FixedLenFeature([], tf.string),
        "img1": tf.FixedLenFeature([], tf.string),
        "mv0": tf.FixedLenFeature([16], tf.float32),
        "mvi0": tf.FixedLenFeature([16], tf.float32),
        "mv1": tf.FixedLenFeature([16], tf.float32),
        "mvi1": tf.FixedLenFeature([16], tf.float32),
    }

    def __init__(self, dataset_dir='datasets/00004/'):
        self.dataset_dir = dataset_dir

        # get the tfrecord filenames
        tfrecord_path = os.path.join(dataset_dir, '*.tfrecord')
        tfrecord_filenames = glob.glob(tfrecord_path)

        def parser(serialized_example):
            fs = tf.parse_single_example(
                serialized_example,
                features=self.feature_set
                # features=self.keypointnet_features
            )
            return fs

        batch_size = 1
        dataset = tf.data.TFRecordDataset(tfrecord_filenames)
        dataset = dataset.map(parser, num_parallel_calls=4)
        dataset = dataset.shuffle(400).repeat().batch(batch_size)
        dataset = dataset.prefetch(buffer_size=256)

        self.iterator = dataset.make_one_shot_iterator().get_next()


# TODO(ethan): make this run in a more elagant manner with argparse values
if __name__ == "__main__":
    # TODO(ethan): get data from argparse

    dataloader = OccnetTfrecordLoader()

    sess = tf.Session()
    
    features = dataloader.iterator

    img0 = tf.image.decode_png(features["img0"][0], 3)
    img0_mask = features["img0_mask"][0]
    img0_depth = features["img0_depth"][0]
    img1 = tf.image.decode_png(features["img1"][0], 3)
    img1_mask = features["img1_mask"][0]
    img1_depth = features["img1_depth"][0]

    img0, img0_mask, img0_depth, img1, img1_mask, img1_depth = sess.run([img0, img0_mask, img0_depth, img1, img1_mask, img1_depth])

    # reshape the stuff
    img0_mask = img0_mask.reshape((128, 128))
    img0_depth = img0_depth.reshape((128, 128))
    img1_mask = img1_mask.reshape((128, 128))
    img1_depth = img1_depth.reshape((128, 128))

    # get into same shape
    mask_copy = np.zeros_like(img0)
    mask_copy[:,:,0] = mask_copy[:,:,1] = mask_copy[:,:,2] = img0_mask
    img0_mask = mask_copy * 255

    mask_copy = np.zeros_like(img1)
    mask_copy[:,:,0] = mask_copy[:,:,1] = mask_copy[:,:,2] = img1_mask
    img1_mask = mask_copy * 255

    # do the same for the depths
    depth_copy = np.zeros_like(img0)
    depth_copy[:,:,0] = depth_copy[:,:,1] = depth_copy[:,:,2] = img0_depth
    img0_depth = depth_copy

    depth_copy = np.zeros_like(img1)
    depth_copy[:,:,0] = depth_copy[:,:,1] = depth_copy[:,:,2] = img1_depth
    img1_depth = depth_copy

    # --------------------
    # a = {}
    # a["img0"] = img0
    # a["img1"] = img1
    # a["mv0"] = mv0
    # a["mvi0"] = mvi0
    # a["mv1"] = mv1
    # a["mvi1"] = mvi1
    # optionally save to a pickle file for loading in notebooks
    # with open('pickled_data.pickle', 'wb') as handle:
    #     pickle.dump(a, handle)
    # --------------------

    # stack images together
    left = np.vstack([img0, img0_mask, img0_depth])
    right = np.vstack([img1, img1_mask, img1_depth])
    image = np.hstack([left, right])

    cv2.imwrite("images/read_tfrecords.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))




