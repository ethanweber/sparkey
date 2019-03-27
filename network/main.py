# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""KeypointNet!!

A reimplementation of 'Discovery of Latent 3D Keypoints via End-to-end
Geometric Reasoning' keypoint network. Given a single 2D image of a known class,
this network can predict a set of 3D keypoints that are consistent across
viewing angles of the same object and across object instances. These keypoints
and their detectors are discovered and learned automatically without
keypoint location supervision.

# ethan
python main.py --model_dir=MODEL_DIR --dset=DSET
example:
(excluding model_dir creates a timestamped model_dir)
python main.py --dset=DSET
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import misc
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
import datetime
import shutil
# ethan: make this import better
import sys
sys.path.append("../")
import network.utils as utils

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean("predict", False, "Running inference if true")
# ethan: adding this to run on original network if desired
tf.app.flags.DEFINE_boolean("keypointnet", False, "Using the keypointnet data if true")
# ethan: adding this to run predictions at different checkpoints
tf.app.flags.DEFINE_string("latest_filename", None, "Name of model.ckpt-# to use when using the --predict flag.")
tf.app.flags.DEFINE_string(
    "input",
    "",
    "Input folder containing images")
tf.app.flags.DEFINE_string("model_dir", None, "Estimator model_dir")
tf.app.flags.DEFINE_string(
    "dset",
    "",
    "Path to the directory containing the dataset.")
tf.app.flags.DEFINE_integer("steps", 200000, "Training steps")
tf.app.flags.DEFINE_integer("batch_size", 8, "Size of mini-batch.")
tf.app.flags.DEFINE_string(
    "hparams", "",
    "A comma-separated list of `name=value` hyperparameter values. This flag "
    "is used to override hyperparameter settings either when manually "
    "selecting hyperparameters or when using Vizier.")
tf.app.flags.DEFINE_integer(
    "sync_replicas", -1,
    "If > 0, use SyncReplicasOptimizer and use this many replicas per sync.")

# Fixed input size 128 x 128.
vw = vh = 128


def create_input_fn(split, batch_size):
  """Returns input_fn for tf.estimator.Estimator.

  Reads tfrecords and construts input_fn for either training or eval. All
  tfrecords not in test.txt or dev.txt will be assigned to training set.

  Args:
    split: A string indicating the split. Can be either 'train' or 'validation'.
    batch_size: The batch size!

  Returns:
    input_fn for tf.estimator.Estimator.

  Raises:
    IOError: If test.txt or dev.txt are not found.
  """

  if (not os.path.exists(os.path.join(FLAGS.dset, "test.txt")) or
      not os.path.exists(os.path.join(FLAGS.dset, "dev.txt"))):
    raise IOError("test.txt or dev.txt not found")

  with open(os.path.join(FLAGS.dset, "test.txt"), "r") as f:
    testset = [x.strip() for x in f.readlines()]

  with open(os.path.join(FLAGS.dset, "dev.txt"), "r") as f:
    validset = [x.strip() for x in f.readlines()]

  files = os.listdir(FLAGS.dset)
  filenames = []
  for f in files:
    sp = os.path.splitext(f)
    if sp[1] != ".tfrecord" or sp[0] in testset:
      continue

    if ((split == "validation" and sp[0] in validset) or
        (split == "train" and sp[0] not in validset)):
      filenames.append(os.path.join(FLAGS.dset, f))

  def input_fn():
    """input_fn for tf.estimator.Estimator."""

    def parser(serialized_example):
      """Parses a single tf.Example into image and label tensors."""
      fs = tf.parse_single_example(
          serialized_example,
          # features={
          #     "img0": tf.FixedLenFeature([], tf.string),
          #     "img1": tf.FixedLenFeature([], tf.string),
          #     "mv0": tf.FixedLenFeature([16], tf.float32),
          #     "mvi0": tf.FixedLenFeature([16], tf.float32),
          #     "mv1": tf.FixedLenFeature([16], tf.float32),
          #     "mvi1": tf.FixedLenFeature([16], tf.float32),
          # })
          # ethan: add more configurable support for occnet
          features={
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
          })

      fs["img0"] = tf.div(tf.to_float(tf.image.decode_png(fs["img0"], 4)), 255)
      fs["img1"] = tf.div(tf.to_float(tf.image.decode_png(fs["img1"], 4)), 255)

      fs["img0"].set_shape([vh, vw, 4])
      fs["img1"].set_shape([vh, vw, 4])

      # fs["lr0"] = [fs["mv0"][0]]
      # fs["lr1"] = [fs["mv1"][0]]

      fs["lr0"] = tf.convert_to_tensor([fs["mv0"][0]])
      fs["lr1"] = tf.convert_to_tensor([fs["mv1"][0]])

      return fs

    np.random.shuffle(filenames)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parser, num_parallel_calls=4)
    dataset = dataset.shuffle(400).repeat().batch(batch_size)
    dataset = dataset.prefetch(buffer_size=256)

    return dataset.make_one_shot_iterator().get_next(), None

  return input_fn


class Transformer(object):
  """A utility for projecting 3D points to 2D coordinates and vice versa.

  3D points are represented in 4D-homogeneous world coordinates. The pixel
  coordinates are represented in normalized device coordinates [-1, 1].
  See https://learnopengl.com/Getting-started/Coordinate-Systems.
  """

  def __get_matrix(self, lines):
    return np.array([[float(y) for y in x.strip().split(" ")] for x in lines])

  def __read_projection_matrix(self, filename):
    if not os.path.exists(filename):
      filename = "/cns/vz-d/home/supasorn/datasets/cars/projection.txt"
    with open(filename, "r") as f:
      lines = f.readlines()
    return self.__get_matrix(lines)

  def __init__(self, w, h, dataset_dir, occnet=False):
    """
    occnet = False for keypointnet data
           = True for occnet data
    """
    self.occnet = occnet
    self.w = w
    self.h = h
    p = self.__read_projection_matrix(dataset_dir + "projection.txt")

    if not self.occnet:
      # keypointnet code
      # transposed of inversed projection matrix.
      self.pinv_t = tf.constant([[1.0 / p[0, 0], 0, 0,
                                  0], [0, 1.0 / p[1, 1], 0, 0], [0, 0, 1, 0],
                                [0, 0, 0, 1]])
      self.f = p[0, 0]
    else:
      # ethan: now this is a 3x3
      # ethan: using the full projection matrix for our data
      self.pinv_t = tf.convert_to_tensor(np.linalg.inv(p), dtype=tf.float32)
      self.p = tf.convert_to_tensor(p, dtype=tf.float32)

  def scale_normalized_coords_to_image_coords(self, normalized_coords):
    """
    Scales the normalized coordinates into the format needed by our projection matrix.
    
    Args:
        normalized_coords: [batch, num_kp, 3] Tensor of keypoints with u and v in the range [-1, 1]
    Returns:
        [batch, num_kp, 3]: Keypoints with u in range [0, 640] and v in range [0, 480]
    """

    # flip the sign of the normalized coordinates
    y_flip_matrix = tf.constant([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    image_coords_y_flipped = tf.matmul(
        tf.reshape(normalized_coords, [-1, normalized_coords.shape[2].value]),
        tf.transpose(y_flip_matrix)
    )
    new_normalized_coords = tf.reshape(
        image_coords_y_flipped,
        [-1, normalized_coords.shape[1].value, normalized_coords.shape[2].value]
    )

    
    scale_matrix = tf.constant([
        [320.0, 0.0, 0.0],
        [0.0, 240.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    offset_matrix = tf.constant([320.0, 240.0, 0.0])
    
    multiplication = tf.matmul(
        tf.reshape(new_normalized_coords, [-1, normalized_coords.shape[2].value]),
        tf.transpose(scale_matrix)
    )
    
    return tf.reshape(
        multiplication + offset_matrix,
        [-1, normalized_coords.shape[1].value, normalized_coords.shape[2].value]
    )

  def scale_image_coords_to_normalized_coords(self, image_coords):
    """
    Scales the image coordinates to normalized coordinates in the range of [-1, 1]
    
    Args:
        image_coords: [batch, num_kp, 3]
    Returns:
        [batch, num_kp, 3]: with normalized u, v values in range [-1, 1]
    """

    # y_flip_matrix = tf.constant([
    #     [1.0, 0.0, 0.0],
    #     [0.0, -1.0, 0.0],
    #     [0.0, 0.0, 1.0]
    # ])
    # image_coords_y_flipped = tf.matmul(
    #     tf.reshape(image_coords, [-1, image_coords.shape[2].value]) + tf.constant([0.0, 480.0, 0.0]),
    #     tf.transpose(y_flip_matrix)
    # )
    # image_coords_back = tf.reshape(
    #     image_coords_y_flipped,
    #     [-1, image_coords.shape[1].value, image_coords.shape[2].value]
    # )

    
    scale_matrix = tf.linalg.inv(
        tf.constant([
            [320.0, 0.0, 0.0],
            [0.0, 240.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
    )
    
    offset_matrix = tf.constant([320.0, 240.0, 0.0])
    
    multiplication = tf.matmul(
        tf.reshape(image_coords, [-1, image_coords.shape[2].value]) - offset_matrix,
        tf.transpose(scale_matrix)
    )
    
    normalized_coords = tf.reshape(
        multiplication,
        [-1, image_coords.shape[1].value, image_coords.shape[2].value]
    )

    # flip the sign of the normalized coordinates
    y_flip_matrix = tf.constant([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    image_coords_y_flipped = tf.matmul(
        tf.reshape(normalized_coords, [-1, image_coords.shape[2].value]),
        tf.transpose(y_flip_matrix)
    )
    return tf.reshape(
        image_coords_y_flipped,
        [-1, image_coords.shape[1].value, image_coords.shape[2].value]
    )


  def project(self, xyzw):
    """
    Projects homogeneous 3D coordinates to normalized device coordinates.
    
    Args:
        xyzw: [batch, num_kp, 4]
    Returns:
        [batch, num_kp, 2]:
    """

    if not self.occnet:
      # keypointnet code
      z = xyzw[:, :, 2:3] + 1e-8
      return tf.concat([-self.f * xyzw[:, :, :2] / z, z], axis=2)

    else:
      # occnet code
      def batch_matmul(a, b):
        return tf.reshape(
            tf.matmul(tf.reshape(a, [-1, a.shape[2].value]), b),
            [-1, a.shape[1].value, a.shape[2].value])

      # the last coordinate of the homogeneous coordinates
      w = xyzw[:, :, 3:]

      # project back into the image frame
      image_coords = batch_matmul(
        xyzw[:, :, :3] / w, tf.transpose(self.p)
      )

      # this will be what we are trying to regress for z
      z = image_coords[:, :, 2:]

      # normalize the uv coordiantes back to [-1, 1] range
      normalized_uv = self.scale_image_coords_to_normalized_coords(image_coords / z)[:, :, :2]

      # put the regressed z value back in place
      normalized_coords = tf.concat([normalized_uv, z], axis=2)

      # ethan: maybe need to handle this later
      # # flip the y value axis to account for projection differences
      # scaler = tf.constant(
      #   [
      #     [1,0,0],
      #     [0,-1,0],
      #     [0,0,1]
      #   ],
      #   dtype=tf.float32
      # )

      # # # no transpose needed here
      # normalized_coords = batch_matmul(normalized_coords, scaler)

      return normalized_coords

  def unproject(self, xyz):
    """
    Unprojects normalized device coordinates with depth to 3D coordinates.
    
    Args:
        xyz: [batch, num_kp, 3]
    Returns:
        [batch, num_kp, 4]:
    """

    if not self.occnet:
      # keypointnet code
      z = xyz[:, :, 2:]
      xy = -xyz * z

      def batch_matmul(a, b):
        return tf.reshape(
            tf.matmul(tf.reshape(a, [-1, a.shape[2].value]), b),
            [-1, a.shape[1].value, a.shape[2].value])

      return batch_matmul(
          tf.concat([xy[:, :, :2], z, tf.ones_like(z)], axis=2), self.pinv_t)

    else:
      # occnet code
      def batch_matmul(a, b):
        return tf.reshape(
            tf.matmul(tf.reshape(a, [-1, a.shape[2].value]), b),
            [-1, a.shape[1].value, a.shape[2].value])

      # get the z values
      z = xyz[:, :, 2:]

      # convert to image coordinates
      uv = self.scale_normalized_coords_to_image_coords(xyz)[:, :, :2]

      # put into camera plane coordinates, which requires multiplying by z
      uvz = tf.concat([uv * z, z], axis=2)

      # multiply by inverse projection matrix and concatenate 1s
      return tf.concat(
        [batch_matmul(uvz, tf.transpose(self.pinv_t)), tf.ones_like(z)], axis=2
      )


def meshgrid(h):
  """Returns a meshgrid ranging from [-1, 1] in x, y axes."""

  r = np.arange(0.5, h, 1) / (h / 2) - 1
  ranx, rany = tf.meshgrid(r, -r)
  return tf.to_float(ranx), tf.to_float(rany)


def estimate_rotation(xyz0, xyz1, pconf, noise):
  """Estimates the rotation between two sets of keypoints.

  The rotation is estimated by first subtracting mean from each set of keypoints
  and computing SVD of the covariance matrix.

  Args:
    xyz0: [batch, num_kp, 3] The first set of keypoints.
    xyz1: [batch, num_kp, 3] The second set of keypoints.
    pconf: [batch, num_kp] The weights used to compute the rotation estimate.
    noise: A number indicating the noise added to the keypoints.

  Returns:
    [batch, 3, 3] A batch of transposed 3 x 3 rotation matrices.
  """

  xyz0 += tf.random_normal(tf.shape(xyz0), mean=0, stddev=noise)
  xyz1 += tf.random_normal(tf.shape(xyz1), mean=0, stddev=noise)

  pconf2 = tf.expand_dims(pconf, 2)
  cen0 = tf.reduce_sum(xyz0 * pconf2, 1, keepdims=True)
  cen1 = tf.reduce_sum(xyz1 * pconf2, 1, keepdims=True)

  x = xyz0 - cen0
  y = xyz1 - cen1

  cov = tf.matmul(tf.matmul(x, tf.matrix_diag(pconf), transpose_a=True), y)
  _, u, v = tf.svd(cov, full_matrices=True)

  d = tf.matrix_determinant(tf.matmul(v, u, transpose_b=True))
  ud = tf.concat(
      [u[:, :, :-1], u[:, :, -1:] * tf.expand_dims(tf.expand_dims(d, 1), 1)],
      axis=2)
  return tf.matmul(ud, v, transpose_b=True)


def relative_pose_loss(xyz0, xyz1, rot, pconf, noise):
  """Computes the relative pose loss (chordal, angular).

  Args:
    xyz0: [batch, num_kp, 3] The first set of keypoints.
    xyz1: [batch, num_kp, 3] The second set of keypoints.
    rot: [batch, 4, 4] The ground-truth rotation matrices.
    pconf: [batch, num_kp] The weights used to compute the rotation estimate.
    noise: A number indicating the noise added to the keypoints.

  Returns:
    A tuple (chordal loss, angular loss).
  """

  r_transposed = estimate_rotation(xyz0, xyz1, pconf, noise)
  rotation = rot[:, :3, :3]
  frob_sqr = tf.reduce_sum(tf.square(r_transposed - rotation), axis=[1, 2])
  frob = tf.sqrt(frob_sqr)

  return tf.reduce_mean(frob_sqr), \
      2.0 * tf.reduce_mean(tf.asin(tf.minimum(1.0, frob / (2 * math.sqrt(2)))))


def separation_loss(xyz, delta):
  """Computes the separation loss.

  Args:
    xyz: [batch, num_kp, 3] Input keypoints.
    delta: A separation threshold. Incur 0 cost if the distance >= delta.

  Returns:
    The seperation loss.
  """

  num_kp = tf.shape(xyz)[1]
  t1 = tf.tile(xyz, [1, num_kp, 1])

  t2 = tf.reshape(tf.tile(xyz, [1, 1, num_kp]), tf.shape(t1))
  diffsq = tf.square(t1 - t2)

  # -> [batch, num_kp ^ 2]
  lensqr = tf.reduce_sum(diffsq, axis=2)

  return (tf.reduce_sum(tf.maximum(-lensqr + delta, 0.0)) / tf.to_float(
      num_kp * FLAGS.batch_size * 2))


def consistency_loss(uv0, uv1, pconf):
  """Computes multi-view consistency loss between two sets of keypoints.

  Args:
    uv0: [batch, num_kp, 2] The first set of keypoint 2D coordinates.
    uv1: [batch, num_kp, 2] The second set of keypoint 2D coordinates.
    pconf: [batch, num_kp] The weights used to compute the rotation estimate.

  Returns:
    The consistency loss.
  """

  # [batch, num_kp, 2]
  wd = tf.square(uv0 - uv1) * tf.expand_dims(pconf, 2)
  wd = tf.reduce_sum(wd, axis=[1, 2])
  return tf.reduce_mean(wd)


def variance_loss(probmap, ranx, rany, uv):
  """Computes the variance loss as part of Sillhouette consistency.

  Args:
    probmap: [batch, num_kp, h, w] The distribution map of keypoint locations.
    ranx: X-axis meshgrid.
    rany: Y-axis meshgrid.
    uv: [batch, num_kp, 2] Keypoint locations (in NDC).

  Returns:
    The variance loss.
  """

  ran = tf.stack([ranx, rany], axis=2)

  sh = tf.shape(ran)
  # [batch, num_kp, vh, vw, 2]
  ran = tf.reshape(ran, [1, 1, sh[0], sh[1], 2])

  sh = tf.shape(uv)
  uv = tf.reshape(uv, [sh[0], sh[1], 1, 1, 2])

  diff = tf.reduce_sum(tf.square(uv - ran), axis=4)
  diff *= probmap

  return tf.reduce_mean(tf.reduce_sum(diff, axis=[2, 3]))

# ethan: created this because we know the depth
def depth_loss(t, uvz, gt_depth):
  """Computes the depth loss.

  Args:
    t: the Transformer instance
    uvz: [batch, num_kp, 3]
    gt_depth: [batch, 307200]

  Returns:
    The depth loss.
  """

  # reshape the map to an image. recall that we use images of size 128 x 128 though
  # img_depth = tf.reshape(gt_depth, [-1, 480, 640])

  # convert to image coordinates with correct dimensions
  xyz = t.scale_normalized_coords_to_image_coords(uvz)

  x = tf.math.floor(xyz[:, :, 0])
  y = tf.math.floor(xyz[:, :, 1])
  z = xyz[:, :, 2]

  # todo: ethan, verify this is correct
  z_index = tf.dtypes.cast(
      tf.clip_by_value(x + y*640, 0.0, 307200-1), 
      tf.int32
  )

  gt_z_values = []
  # TODO(ethan): fix this batch calculation! should not set it here to 8!
  # for i in range(z.shape[0]):
  for i in range(8):
      gt_z_values.append(
          tf.gather(
              gt_depth[i] / 1000.0,
              z_index[i]
          )
      )
      
  gt_z_values = tf.stack(gt_z_values)

  error = tf.losses.mean_squared_error(z, gt_z_values)

  return error


def dilated_cnn(images, num_filters, is_training):
  """Constructs a base dilated convolutional network.

  Args:
    images: [batch, h, w, 3] Input RGB images.
    num_filters: The number of filters for all layers.
    is_training: True if this function is called during training.

  Returns:
    Output of this dilated CNN.
  """
  #ethan: doesn't images have a 4th channeL?

  net = images

  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      normalizer_fn=slim.batch_norm,
      activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
      normalizer_params={"is_training": is_training}):
    for i, r in enumerate([1, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1]):
      net = slim.conv2d(net, num_filters, [3, 3], rate=r, scope="dconv%d" % i)

  return net


def orientation_network(images, num_filters, is_training):
  """Constructs a network that infers the orientation of an object.

  Args:
    images: [batch, h, w, 3] Input RGB images.
    num_filters: The number of filters for all layers.
    is_training: True if this function is called during training.

  Returns:
    Output of the orientation network.
  """

  with tf.variable_scope("OrientationNetwork"):
    net = dilated_cnn(images, num_filters, is_training)

    modules = 2
    prob = slim.conv2d(net, 2, [3, 3], rate=1, activation_fn=None)
    prob = tf.transpose(prob, [0, 3, 1, 2])

    prob = tf.reshape(prob, [-1, modules, vh * vw])
    prob = tf.nn.softmax(prob)
    ranx, rany = meshgrid(vh)

    prob = tf.reshape(prob, [-1, 2, vh, vw])

    sx = tf.reduce_sum(prob * ranx, axis=[2, 3])
    sy = tf.reduce_sum(prob * rany, axis=[2, 3])  # -> batch x modules

    out_xy = tf.reshape(tf.stack([sx, sy], -1), [-1, modules, 2])

  return out_xy


def keypoint_network(rgba,
                     num_filters,
                     num_kp,
                     is_training,
                     lr_gt=None,
                     anneal=1):
  """Constructs our main keypoint network that predicts 3D keypoints.

  Args:
    rgba: [batch, h, w, 4] Input RGB images with alpha channel.
    num_filters: The number of filters for all layers.
    num_kp: The number of keypoints.
    is_training: True if this function is called during training.
    lr_gt: The groundtruth orientation flag used at the beginning of training.
        Then we linearly anneal in the prediction.
    anneal: A number between [0, 1] where 1 means using the ground-truth
        orientation and 0 means using our estimate.

  Returns:
    uv: [batch, num_kp, 2] 2D locations of keypoints.
    z: [batch, num_kp] The depth of keypoints.
    orient: [batch, 2, 2] Two 2D coordinates that correspond to [1, 0, 0] and
        [-1, 0, 0] in object space.
    sill: The Sillhouette loss.
    variance: The variance loss.
    prob_viz: A visualization of all predicted keypoints.
    prob_vizs: A list of visualizations of each keypoint.

  """

  images = rgba[:, :, :, :3]

  # [batch, 1]
  orient = orientation_network(images, num_filters * 0.5, is_training)

  # [batch, 1]
  lr_estimated = tf.maximum(0.0, tf.sign(orient[:, 0, :1] - orient[:, 1, :1]))

  if lr_gt is None:
    lr = lr_estimated
  else:
    lr_gt = tf.maximum(0.0, tf.sign(lr_gt[:, :1]))
    lr = tf.round(lr_gt * anneal + lr_estimated * (1 - anneal))

  lrtiled = tf.tile(
      tf.expand_dims(tf.expand_dims(lr, 1), 1),
      [1, images.shape[1], images.shape[2], 1])

  images = tf.concat([images, lrtiled], axis=3)

  mask = rgba[:, :, :, 3]
  mask = tf.cast(tf.greater(mask, tf.zeros_like(mask)), dtype=tf.float32)

  net = dilated_cnn(images, num_filters, is_training)

  # The probability distribution map.
  prob = slim.conv2d(
      net, num_kp, [3, 3], rate=1, scope="conv_xy", activation_fn=None)

  # ethan: figure out what our fixed camera distance might be. keypointnet used -30
  # We added the  fixed camera distance as a bias.
  # z = -30 + slim.conv2d(
  #     net, num_kp, [3, 3], rate=1, scope="conv_z", activation_fn=None)
  z = slim.conv2d(
    net, num_kp, [3, 3], rate=1, scope="conv_z", activation_fn=None)

  prob = tf.transpose(prob, [0, 3, 1, 2])
  z = tf.transpose(z, [0, 3, 1, 2])

  prob = tf.reshape(prob, [-1, num_kp, vh * vw])
  prob = tf.nn.softmax(prob, name="softmax")

  ranx, rany = meshgrid(vh)
  prob = tf.reshape(prob, [-1, num_kp, vh, vw])

  # These are for visualizing the distribution maps.
  prob_viz = tf.expand_dims(tf.reduce_sum(prob, 1), 3)
  prob_vizs = [tf.expand_dims(prob[:, i, :, :], 3) for i in range(num_kp)]

  sx = tf.reduce_sum(prob * ranx, axis=[2, 3])
  sy = tf.reduce_sum(prob * rany, axis=[2, 3])  # -> batch x num_kp

  # [batch, num_kp]
  sill = tf.reduce_sum(prob * tf.expand_dims(mask, 1), axis=[2, 3])
  sill = tf.reduce_mean(-tf.log(sill + 1e-12))

  z = tf.reduce_sum(prob * z, axis=[2, 3])
  uv = tf.reshape(tf.stack([sx, sy], -1), [-1, num_kp, 2])

  variance = variance_loss(prob, ranx, rany, uv)

  return uv, z, orient, sill, variance, prob_viz, prob_vizs


def model_fn(features, labels, mode, hparams):
  """Returns model_fn for tf.estimator.Estimator."""

  del labels

  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  # TODO(ethan): need to change the occnet flag later
  t = Transformer(vw, vh, FLAGS.dset, occnet=not FLAGS.keypointnet)

  def func1(x):
    return tf.transpose(tf.reshape(features[x], [-1, 4, 4]), [0, 2, 1])

  mv = [func1("mv%d" % i) for i in range(2)]
  mvi = [func1("mvi%d" % i) for i in range(2)]

  uvz = [None] * 2
  uvz_proj = [None] * 2  # uvz coordinates projected on to the other view.
  viz = [None] * 2
  vizs = [None] * 2

  loss_sill = 0
  loss_variance = 0
  loss_con = 0
  loss_sep = 0
  loss_lr = 0
  loss_depth = 0 # ethan adding this because we have ground truth depth

  for i in range(2):
    with tf.variable_scope("KeypointNetwork", reuse=i > 0):

      # anneal: 1 = using ground-truth, 0 = using our estimate orientation.
      anneal = tf.to_float(hparams.lr_anneal_end - tf.train.get_global_step())
      anneal = tf.clip_by_value(
          anneal / (hparams.lr_anneal_end - hparams.lr_anneal_start), 0.0, 1.0)

      uv, z, orient, sill, variance, viz[i], vizs[i] = keypoint_network(
          features["img%d" % i],
          hparams.num_filters,
          hparams.num_kp,
          is_training,
          lr_gt=features["lr%d" % i],
          anneal=anneal)

      # x-positive/negative axes (dominant direction).
      xp_axis = tf.tile(
          tf.constant([[[1.0, 0, 0, 1], [-1.0, 0, 0, 1]]]),
          [tf.shape(orient)[0], 1, 1])

      # [batch, 2, 4]  = [batch, 2, 4] x [batch, 4, 4]
      xp = tf.matmul(xp_axis, mv[i])

      # [batch, 2, 3]
      xp = t.project(xp)

      loss_lr += tf.losses.mean_squared_error(orient[:, :, :2], xp[:, :, :2])
      loss_variance += variance
      loss_sill += sill

      uv = tf.reshape(uv, [-1, hparams.num_kp, 2])
      z = tf.reshape(z, [-1, hparams.num_kp, 1])

      # [batch, num_kp, 3]
      uvz[i] = tf.concat([uv, z], axis=2)

      world_coords = tf.matmul(t.unproject(uvz[i]), mvi[i])

      # [batch, num_kp, 3]
      uvz_proj[i] = t.project(tf.matmul(world_coords, mv[1 - i]))

  pconf = tf.ones(
      [tf.shape(uv)[0], tf.shape(uv)[1]], dtype=tf.float32) / hparams.num_kp

  depth_multiplier = tf.to_float(hparams.remove_depth_start_pose - tf.train.get_global_step())
  depth_multiplier = tf.clip_by_value(
    depth_multiplier / hparams.remove_depth_start_pose, 0.0, 1.0)
  pose_multiplier = 1.0 - depth_multiplier

  for i in range(2):
    loss_con += consistency_loss(uvz_proj[i][:, :, :2], uvz[1 - i][:, :, :2],
                                 pconf)
    loss_sep += separation_loss(
        t.unproject(uvz[i])[:, :, :3], hparams.sep_delta)
    loss_depth += depth_loss(
      t,
      uvz[i],
      features["img%d_depth" % i]
    )

  chordal, angular = relative_pose_loss(
      t.unproject(uvz[0])[:, :, :3],
      t.unproject(uvz[1])[:, :, :3], tf.matmul(mvi[0], mv[1]), pconf,
      hparams.noise)

  loss = (
      hparams.loss_pose * angular+#*pose_multiplier +
      hparams.loss_con * loss_con +
      hparams.loss_sep * loss_sep +
      hparams.loss_sill * loss_sill +
      hparams.loss_lr * loss_lr +
      hparams.loss_variance * loss_variance +
      hparams.loss_depth * loss_depth#*depth_multiplier
  )

  def touint8(img):
    return tf.cast(img * 255.0, tf.uint8)

  with tf.variable_scope("output"):
    tf.summary.image("0_img0", touint8(features["img0"][:, :, :, :3]))
    tf.summary.image("1_combined", viz[0])
    for i in range(hparams.num_kp):
      tf.summary.image("2_f%02d" % i, vizs[0][i])

  with tf.variable_scope("stats"):
    tf.summary.scalar("anneal", anneal)
    tf.summary.scalar("closs", loss_con)
    tf.summary.scalar("seploss", loss_sep)
    tf.summary.scalar("angular", angular)
    tf.summary.scalar("chordal", chordal)
    tf.summary.scalar("lrloss", loss_lr)
    tf.summary.scalar("sill", loss_sill)
    tf.summary.scalar("vloss", loss_variance)
    tf.summary.scalar("dloss", loss_depth)
    tf.summary.scalar("pose_multiplier", pose_multiplier)
    tf.summary.scalar("depth_multiplier", depth_multiplier)

  return {
      "loss": loss,
      "predictions": {
          "img0": features["img0"],
          "img1": features["img1"],
          "uvz0": uvz[0],
          "uvz1": uvz[1]
      },
      "eval_metric_ops": {
          "closs": tf.metrics.mean(loss_con),
          "angular_loss": tf.metrics.mean(angular),
          "chordal_loss": tf.metrics.mean(chordal),
      }
  }


def predict(input_folder, hparams):
  """Predicts keypoints on all images in input_folder."""

  cols = plt.cm.get_cmap("rainbow")(
      np.linspace(0, 1.0, hparams.num_kp))[:, :4]

  img = tf.placeholder(tf.float32, shape=(1, 128, 128, 4))

  with tf.variable_scope("KeypointNetwork"):
    ret = keypoint_network(
        img, hparams.num_filters, hparams.num_kp, False)

  uv = tf.reshape(ret[0], [-1, hparams.num_kp, 2])
  z = tf.reshape(ret[1], [-1, hparams.num_kp, 1])
  uvz = tf.concat([uv, z], axis=2)

  sess = tf.Session()
  saver = tf.train.Saver()
  
  # ethan: add a paramter to set the checkpoint
  # create the file if FLAGS.latest_filename isn't None. rewrite it if it already exists
  if FLAGS.latest_filename is not None:
    latest_filename_path = os.path.join(FLAGS.model_dir, FLAGS.latest_filename)
    f = open(latest_filename_path, "w")
    f.write("model_checkpoint_path: \"{}\"".format(FLAGS.latest_filename))
    f.close()
  ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir, latest_filename=FLAGS.latest_filename)

  print("loading model: ", ckpt.model_checkpoint_path)
  saver.restore(sess, ckpt.model_checkpoint_path)

  files = [x for x in os.listdir(input_folder)
           if x[-3:] in ["jpg", "png"]]

  output_folder = os.path.join(input_folder, "output")
  if not os.path.exists(output_folder):
    os.mkdir(output_folder)

  for f in files:
    orig = misc.imread(os.path.join(input_folder, f)).astype(float) / 255
    if orig.shape[2] == 3:
      orig = np.concatenate((orig, np.ones_like(orig[:, :, :1])), axis=2)

    uv_ret = sess.run(uvz, feed_dict={img: np.expand_dims(orig, 0)})

    utils.draw_ndc_points(orig, uv_ret.reshape(hparams.num_kp, 3), cols)
    misc.imsave(os.path.join(output_folder, f), orig)


def _default_hparams():
  """Returns default or overridden user-specified hyperparameters."""

  hparams = tf.contrib.training.HParams(
      num_filters=64,  # Number of filters.
      num_kp=10,  # Numer of keypoints.

      loss_pose=0.0, # ethan
      # loss_pose=0.2,  # Pose Loss.
      loss_con=1.0,  # Multiview consistency Loss.
      loss_sep=0.0, # ethan
      # loss_sep=2.0,  # Seperation Loss.
      loss_sill=1.0,  # Sillhouette Loss.
      loss_lr=0.0,  # ethan
      # loss_lr=1.0,  # Orientation Loss.
      loss_variance=0.0,  # ethan
      # loss_variance=0.5, # Variance Loss (part of Sillhouette loss).
      # ethan: added this because we have gt depth
      # loss_depth=0.0,
      loss_depth=1.0, # Depth Loss
      # when to remove depth from loss
      remove_depth_start_pose=10000,

      # sep_delta=0.05,  # Seperation threshold.
      sep_delta=0.002,  # ethan: seperation loss with our data. should be smaller I think
      noise=0.1,  # Noise added during estimating rotation.

      learning_rate=1.0e-3,
      lr_anneal_start=30000,  # When to anneal in the orientation prediction.
      lr_anneal_end=60000,  # When to use the prediction completely.
  )
  if FLAGS.hparams:
    hparams = hparams.parse(FLAGS.hparams)
  return hparams


def main(argv):
  del argv

  hparams = _default_hparams()


  if FLAGS.predict:
    predict(FLAGS.input, hparams)
  else:
    # ethan: if no FLAGS.model_dir, then create a new one with a timestamp
    if FLAGS.model_dir is None:
      # get the current timestamp
      timestamp = datetime.datetime.now().strftime("%y-%m-%d_%I:%M:%S")
      current_path = os.path.dirname(os.path.abspath(__file__))
      experiments_path = os.path.join(current_path, "../experiments")
      
      # make the directory, which should not already exist
      actual_model_dir = os.path.join(experiments_path, timestamp)
      if not os.path.exists(actual_model_dir):
        os.makedirs(actual_model_dir)
    else:
      actual_model_dir = FLAGS.model_dir

    # ethan: save a copy of the network/ directory in the experiment to remember what was used for the experiment
    # ethan: make this better in the future!
    current_file = os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "main.py"
    )
    target_file = os.path.join(
      actual_model_dir,
      "main.py"
    )
    shutil.copy(current_file, target_file)

    utils.train_and_eval(
        model_dir=actual_model_dir,
        model_fn=model_fn,
        input_fn=create_input_fn,
        hparams=hparams,
        steps=FLAGS.steps,
        batch_size=FLAGS.batch_size,
        save_checkpoints_secs=600,
        eval_throttle_secs=1800,
        eval_steps=5,
        sync_replicas=FLAGS.sync_replicas,
    )


if __name__ == "__main__":
  sys.excepthook = utils.colored_hook(
      os.path.dirname(os.path.realpath(__file__)))
  tf.app.run()
