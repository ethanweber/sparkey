"""
This file create tfrecord dataset in the correct format for keypointnet.

We start by creating the dataset of just randomly drawn pairs image pairs from a dense object nets dataset.
"""

# TODO(ethan): we currently make the assumption that all data pairs share the same camera intrinsic matrix K

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

from data_loader import OccNetDataloader
import utils as occnet_utils

# TODO(ethan): use these flags
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_image_pairs", 10, "The number of image pairs to use.")
tf.app.flags.DEFINE_string("output", "", "Output tfrecord.")

dataloader = OccNetDataloader()

# code for formatting into tfrecord format

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def generate():
    with tf.python_io.TFRecordWriter(FLAGS.output) as tfrecord_writer:
        
        with tf.Graph().as_default():
            # placeholders for the images
            im0 = tf.placeholder(dtype=tf.uint8)
            im1 = tf.placeholder(dtype=tf.uint8)
            encoded0 = tf.image.encode_png(im0)
            encoded1 = tf.image.encode_png(im1)

            # placeholders for the masks
            im0_mask = tf.placeholder(dtype=tf.uint8)
            im1_mask = tf.placeholder(dtype=tf.uint8)
            encoded0_mask = tf.image.encode_png(im0_mask)
            encoded1_mask = tf.image.encode_png(im1_mask)

            with tf.Session() as sess:

                for i in range(FLAGS.num_image_pairs):

                    # get a data pair and unpack it
                    # TODO(ethan): be careful about how far the image pairs are apart
                    K, a_image_data, b_image_data = dataloader.get_random_data_pair()
                    rgb_a, depth_a, mask_a, pose_a = a_image_data
                    rgb_b, depth_b, mask_b, pose_b = b_image_data
                    
                    # resize images to 128 x 128
                    rgb_a = rgb_a.resize((128, 128))
                    rgb_b = rgb_b.resize((128, 128))

                    # resize masks to 128 x 128
                    mask_a = mask_a.resize((128, 128))
                    mask_b = mask_b.resize((128, 128))

                    # convert the images to numpy arrays
                    image0 = np.array(rgb_a)
                    image1 = np.array(rgb_b)

                    # convert the masks to numpy arrays
                    mask0 = np.zeros_like(image0)
                    mask0[:,:,0] = mask0[:,:,1] = mask0[:,:,2] = np.array(mask_a)
                    mask1 = np.zeros_like(image1)
                    mask1[:,:,0] = mask1[:,:,1] = mask1[:,:,2] = np.array(mask_b)

                    # TODO(ethan): only operate on segmented masks right now
                    # maybe change this later
                    image0 = image0 * mask0
                    image1 = image1 * mask1

                    # camera poses in world frame
                    mat0 = np.array(pose_a)
                    mat1 = np.array(pose_b)
                    mati0 = np.linalg.inv(mat0).flatten()
                    mati1 = np.linalg.inv(mat1).flatten()
                    mat0 = mat0.flatten()
                    mat1 = mat1.flatten()

                    # feed the placeholders for the images and masks
                    st0, st1 = sess.run([encoded0, encoded1],
                        feed_dict={im0: image0, im1: image1})
                    st0_mask, st1_mask = sess.run([encoded0_mask, encoded1_mask],
                        feed_dict={im0_mask: mask0, im1_mask: mask1})

                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'img0': bytes_feature(st0),
                                'img0_mask': bytes_feature(st0_mask),
                                'img1': bytes_feature(st1),
                                'img1_mask': bytes_feature(st1_mask),
                                'mv0': tf.train.Feature(
                                    float_list=tf.train.FloatList(value=mat0)),
                                'mvi0': tf.train.Feature(
                                    float_list=tf.train.FloatList(value=mati0)),
                                'mv1': tf.train.Feature(
                                    float_list=tf.train.FloatList(value=mat1)),
                                'mvi1': tf.train.Feature(
                                    float_list=tf.train.FloatList(value=mati1)),
                            }
                        )
                    )

                    tfrecord_writer.write(example.SerializeToString())


def main(argv):
    del argv
    generate()


# python create_tfrecords.py --output=tfrecords/output.tfrecord
if __name__ == "__main__":
    tf.app.run()