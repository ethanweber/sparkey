"""
This file create tfrecord dataset in the correct format for keypointnet.

We start by creating the dataset of just randomly drawn pairs image pairs from a dense object nets dataset.
"""
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

from data_loader import OccNetDataloader
import utils as occnet_utils

# TODO(ethan): use these flags
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_image_pairs", 100, "The number of image pairs to use.")
tf.app.flags.DEFINE_string("output", "", "Output tfrecord.")

dataloader = OccNetDataloader()

# get a data pair and unpack it
K, a_image_data, b_image_data = dataloader.get_random_data_pair()
rgb_a, depth_a, mask_a, pose_a = a_image_data
rgb_b, depth_b, mask_b, pose_b = b_image_data

# specify the transforms
T_w_a = pose_a
T_w_b = pose_b
T_a_b = np.matmul(np.linalg.inv(T_w_a), T_w_b)
T_b_a = np.matmul(np.linalg.inv(T_w_b), T_w_a)

# get a set of valid coordinates
valid_coords = occnet_utils.get_coords_contained_in_mask(mask_a)
u, v = random.choice(valid_coords) # choose a random, valid coord to use
z_at_u_v = np.array(depth_a)[v,u] / 1000.0
print("Z: {}".format(z_at_u_v))

camera_plane_coord = np.array([u, v, 1])*z_at_u_v
print("Camera plane coord: {}".format(camera_plane_coord))
camera_world_coord = np.matmul(np.linalg.inv(K), camera_plane_coord)
print("Camera world coord: {}".format(camera_world_coord))

# convert to homogeneous coordinates
homogeneous_camera_world_coord = np.concatenate((camera_world_coord, np.ones(1)))
# transform from a to b
world_in_b = np.matmul(T_b_a, homogeneous_camera_world_coord)

# project the world coordinate back into the image plane
# convert back to non homogenous
world_in_b /= world_in_b[3]
print("world in b: {}".format(world_in_b))
projected_into_camera_b = np.matmul(K, world_in_b[:3])
print(projected_into_camera_b / projected_into_camera_b[2])
u_new, v_new, _ = projected_into_camera_b / projected_into_camera_b[2]


a_draw = cv2.circle(np.array(rgb_a), (int(u), int(v)), 10, (255,0,0), -1)
plt.imshow(a_draw)
plt.savefig('images/a_image.png')

b_draw = cv2.circle(np.array(rgb_b), (int(u_new), int(v_new)), 10, (255,0,0), -1)
plt.imshow(b_draw)
plt.savefig('images/b_image.png')

# code for formatting into tfrecord format

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def generate():
    with tf.python_io.TFRecordWriter(FLAGS.output) as tfrecord_writer:
        
        with tf.Graph().as_default():
            im0 = tf.placeholder(dtype=tf.uint8)
            im1 = tf.placeholder(dtype=tf.uint8)
            encoded0 = tf.image.encode_png(im0)
            encoded1 = tf.image.encode_png(im1)

            with tf.Session() as sess:
                # count = 0
                # indir = FLAGS.input + "/"
                # while tf.gfile.Exists(indir + "%06d.txt" % count):
                # print("saving %06d" % count)
                image0 = np.array(rgb_a) # misc.imread(indir + "%06d.png" % (count * 2))
                image1 = np.array(rgb_b) # misc.imread(indir + "%06d.png" % (count * 2 + 1))

                # mat0, mat1 = read_model_view_matrices(indir + "%06d.txt" % count)

                # mati0 = np.linalg.inv(mat0).flatten()
                # mati1 = np.linalg.inv(mat1).flatten()
                # mat0 = mat0.flatten()
                # mat1 = mat1.flatten()

                st0, st1 = sess.run([encoded0, encoded1],
                    feed_dict={im0: image0, im1: image1})

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'img0': bytes_feature(st0),
                            'img1': bytes_feature(st1),
                            # 'mv0': tf.train.Feature(
                            #     float_list=tf.train.FloatList(value=mat0)),
                            # 'mvi0': tf.train.Feature(
                            #     float_list=tf.train.FloatList(value=mati0)),
                            # 'mv1': tf.train.Feature(
                            #     float_list=tf.train.FloatList(value=mat1)),
                            # 'mvi1': tf.train.Feature(
                            #     float_list=tf.train.FloatList(value=mati1)),
                        }
                    )
                )

                tfrecord_writer.write(example.SerializeToString())
                # count += 1


def main(argv):
    del argv
    generate()


# python create_tfrecords.py --output=tfrecords/output.tfrecord
if __name__ == "__main__":
    tf.app.run()