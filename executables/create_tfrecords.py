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

from data_loader import OccnetDataLoader
import utils as occnet_utils

# TODO(ethan): use these flags
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_image_pairs", 10, "The number of image pairs to use.")
tf.app.flags.DEFINE_string("output", "", "Output tfrecord.")

dataloader = OccnetDataLoader()

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

                    # resize depth images to 128 x 128
                    # depth_a = depth_a.resize((128, 128))
                    # depth_b = depth_b.resize((128, 128))

                    # convert the images to numpy arrays
                    image0 = np.array(rgb_a)
                    image1 = np.array(rgb_b)

                    # convert the masks to numpy arrays
                    mask0 = np.array(mask_a)
                    mask1 = np.array(mask_b)

                    # for masking with the images
                    temp_mask0 = np.zeros_like(image0)
                    temp_mask0[:,:,0] = temp_mask0[:,:,1] = temp_mask0[:,:,2] = mask0
                    temp_mask1 = np.zeros_like(image1)
                    temp_mask1[:,:,0] = temp_mask1[:,:,1] = temp_mask1[:,:,2] = mask1

                    # convert the depth maps to numpy arrays
                    depth0 = np.array(depth_a)
                    depth1 = np.array(depth_b)

                    # TODO(ethan): only operate on segmented masks right now
                    # maybe change this later
                    image0 = image0 * temp_mask0
                    image1 = image1 * temp_mask1

                    # expand the images to have alpha channels
                    image0 = np.concatenate((image0, temp_mask0[:,:,:1]), axis=2)
                    image1 = np.concatenate((image1, temp_mask1[:,:,:1]), axis=2)

                    # write these to file
                    # cv2.imwrite("/home/ethanweber/Documents/occnet/caterpillars/{}.png".format(i), image0[...,::-1])

                    # camera poses in world frame
                    # mat0 = np.array(pose_a)
                    # mat1 = np.array(pose_b)
                    # mati0 = np.linalg.inv(mat0).flatten()
                    # mati1 = np.linalg.inv(mat1).flatten()
                    # mat0 = mat0.flatten()
                    # mat1 = mat1.flatten()
                    mati0 = np.array(pose_a)
                    mati1 = np.array(pose_b)
                    mat0 = np.linalg.inv(mati0).flatten()
                    mat1 = np.linalg.inv(mati1).flatten()
                    mati0 = mati0.flatten()
                    mati1 = mati1.flatten()
                    

                    # feed the placeholders for the images, masks and depths
                    st0, st1 = sess.run([encoded0, encoded1],
                        feed_dict={im0: image0, im1: image1})

                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'img0': bytes_feature(st0),
                                'img0_mask': tf.train.Feature(
                                    float_list=tf.train.FloatList(value=mask0.flatten())),
                                'img0_depth': tf.train.Feature(
                                    float_list=tf.train.FloatList(value=depth0.flatten())),
                                'img1': bytes_feature(st1),
                                'img1_mask': tf.train.Feature(
                                    float_list=tf.train.FloatList(value=mask1.flatten())),
                                'img1_depth': tf.train.Feature(
                                    float_list=tf.train.FloatList(value=depth1.flatten())),
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