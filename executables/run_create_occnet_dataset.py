""" executable to creating occnet datasets

args:
 - dataset_name:
 - num_data_points_per_record:
 - num_train:
 - num_test:
 - num_validation:

creates a dataset with `dataset_name` in the `datasets` folder

example:
python run_create_occnet_dataset.py --dataset_name 000 --num_data_points_per_record 100 ...
"""

# ethan: we currently make the assumption that all data pairs share the same camera intrinsic matrix K

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import os
import shutil

# ethan: make this import better
import sys
sys.path.append("../")
from data.don_data_loader import DonDataLoader

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("dataset_name", "999", "Name of the created dataset.")
tf.app.flags.DEFINE_integer("num_data_points_per_record", 10, "The number of image pairs (data points) per tfrecord.")
tf.app.flags.DEFINE_integer("num_train", 6, "The number of image pairs to use.")
tf.app.flags.DEFINE_integer("num_test", 2, "The number of image pairs to use.")
tf.app.flags.DEFINE_integer("num_validation", 2, "The number of image pairs to use.")

dataloader = DonDataLoader()

# code below for writing to a tfrecord

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def generate(tfrecord_filename, num_data_points):

    with tf.python_io.TFRecordWriter(tfrecord_filename) as tfrecord_writer:
        
        with tf.Graph().as_default():
            # placeholders for the images
            im0 = tf.placeholder(dtype=tf.uint8)
            im1 = tf.placeholder(dtype=tf.uint8)
            encoded0 = tf.image.encode_png(im0)
            encoded1 = tf.image.encode_png(im1)

            with tf.Session() as sess:

                for i in range(num_data_points):

                    # get a data pair and unpack it
                    # ethan: maybe be careful about how far apart the image pairs could be
                    K, a_image_data, b_image_data, scene_name = dataloader.get_random_data_pair()
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

                    # ethan: we are only operating on masked images right now
                    # ethan: maybe change this later because right now we must have instance segmentation
                    image0 = image0 * temp_mask0
                    image1 = image1 * temp_mask1

                    # expand the images to have alpha channels, which are the masks
                    image0 = np.concatenate((image0, temp_mask0[:,:,:1]), axis=2)
                    image1 = np.concatenate((image1, temp_mask1[:,:,:1]), axis=2)

                    # ethan: although the values seem flipped, this formats it correctly in keypointnet form (based on experiments)
                    # camera poses in world frame
                    mati0 = np.array(pose_a)
                    mati1 = np.array(pose_b)
                    mat0 = np.linalg.inv(mati0).flatten()
                    mat1 = np.linalg.inv(mati1).flatten()
                    mati0 = mati0.flatten()
                    mati1 = mati1.flatten()

                    centroid = np.array([dataloader.centroid_and_radius[scene_name]["centroid"]])
                    radius = np.array([dataloader.centroid_and_radius[scene_name]["radius"]])

                    # feed the placeholders for the images
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
                                'centroid': tf.train.Feature(
                                    float_list=tf.train.FloatList(value=centroid)),
                                'radius': tf.train.Feature(
                                    float_list=tf.train.FloatList(value=radius)),
                            }
                        )
                    )

                    tfrecord_writer.write(example.SerializeToString())



current_path = os.path.dirname(os.path.abspath(__file__))
datasets_path = os.path.join(current_path, "../datasets")

def main(argv):
    del argv

    new_dataset_path = os.path.join(datasets_path, FLAGS.dataset_name)

    # delete the folder it already exists and create the dataset folder if it doesn't exist
    if os.path.exists(new_dataset_path):
        shutil.rmtree(new_dataset_path, ignore_errors=True)
        os.makedirs(new_dataset_path)
    else:
        os.makedirs(new_dataset_path)

    # holds .txt filenames to be written later
    text_file_dictionary = {}
    text_file_dictionary["train.txt"] = [] # train
    text_file_dictionary["test.txt"] = [] # test
    text_file_dictionary["dev.txt"] = [] # validation

    current_index = 0

    for i in range(FLAGS.num_train):
        filename_index = "{:05d}".format(current_index)
        text_file_dictionary["train.txt"].append(filename_index)

        tfrecord_filename = os.path.join(new_dataset_path, "{}.tfrecord".format(filename_index))
        num_data_points = FLAGS.num_data_points_per_record

        # write the tfrecord
        generate(tfrecord_filename, num_data_points)

        current_index += 1

    for i in range(FLAGS.num_test):
        filename_index = "{:05d}".format(current_index)
        text_file_dictionary["test.txt"].append(filename_index)

        tfrecord_filename = os.path.join(new_dataset_path, "{}.tfrecord".format(filename_index))
        num_data_points = FLAGS.num_data_points_per_record

        # write the tfrecord
        generate(tfrecord_filename, num_data_points)

        current_index += 1

    for i in range(FLAGS.num_validation):
        filename_index = "{:05d}".format(current_index)
        text_file_dictionary["dev.txt"].append(filename_index)

        tfrecord_filename = os.path.join(new_dataset_path, "{}.tfrecord".format(filename_index))
        num_data_points = FLAGS.num_data_points_per_record

        # write the tfrecord
        generate(tfrecord_filename, num_data_points)

        current_index += 1

    # write the .txt files
    for text_file in text_file_dictionary.keys():
        f = open(os.path.join(new_dataset_path, text_file), "w+")
        for value in text_file_dictionary[text_file]:
            f.write(value + "\n")
        f.close()

    # write the instrinsics file
    K, _, _, _ = dataloader.get_random_data_pair()
    f = open(os.path.join(new_dataset_path, "projection.txt"), "w+")
    for row in K:
        row_length = len(row)
        for i in range(row_length):
            f.write(str(row[i]))
            if i < len(row) - 1:
                f.write(" ")
            else:
                f.write("\n")
    f.close()

# see the top of the file for parameters
if __name__ == "__main__":
    tf.app.run()