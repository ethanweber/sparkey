import tensorflow as tf
import cv2
import numpy as np
import glob

# Read and print data:
sess = tf.InteractiveSession()

reader = tf.TFRecordReader()
# filenames = glob.glob('../data/cars_with_keypoints/*.tfrecord')
filenames = glob.glob('tfrecords/*.tfrecord')

filename_queue = tf.train.string_input_producer(filenames)
_, serialized_example = reader.read(filename_queue)

feature_set = {
    "img0": tf.FixedLenFeature([], tf.string),
    "img1": tf.FixedLenFeature([], tf.string),
    # "mv0": tf.FixedLenFeature([16], tf.float32),
    # "mvi0": tf.FixedLenFeature([16], tf.float32),
    # "mv1": tf.FixedLenFeature([16], tf.float32),
    # "mvi1": tf.FixedLenFeature([16], tf.float32),
}

def get_features():
    global serialized_example
    global feature_set

    features = tf.parse_single_example(serialized_example, features=feature_set)
    return features

# Many tf.train functions use tf.train.QueueRunner,
# so we need to start it before we read
# https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564
tf.train.start_queue_runners(sess)

# print all the features
# for name, tensor in features.items():
#     print('{}: {}'.format(name, tensor.eval()))

# evaluate into a numpy array, which will be RGB
# img0 = tf.image.decode_png(features["img0"], 3).eval()
# img0 = tf.div(tf.to_float(tf.image.decode_png(features["img0"], 4)), 255).eval()

# print(img0.shape)
# print(img0.dtype)

# number of images to look at
for i in range(1):

    features = get_features()

    img0 = tf.image.decode_png(features["img0"], 3).eval()
    img1 = tf.image.decode_png(features["img1"], 3).eval()

    # stack the images next to each other
    image = np.hstack([img0, img1])

    # # save the image to visualize
    filename = "images/example_{}.png".format(i)
    cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))








