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
        "img1": tf.FixedLenFeature([], tf.string),
        "mv0": tf.FixedLenFeature([16], tf.float32),
        "mvi0": tf.FixedLenFeature([16], tf.float32),
        "mv1": tf.FixedLenFeature([16], tf.float32),
        "mvi1": tf.FixedLenFeature([16], tf.float32),
    }

    def __init__(self, dataset_dir='datasets/00001/'):
        self.dataset_dir = dataset_dir

        # get the tfrecord filenames
        tfrecord_path = os.path.join(dataset_dir, '*.tfrecord')
        tfrecord_filenames = glob.glob(tfrecord_path)

        self.reader = tf.TFRecordReader()

        filename_queue = tf.train.string_input_producer(tfrecord_filenames)
        _, self.serialized_example = self.reader.read(filename_queue)

    def get_features(self):
        features = tf.parse_single_example(self.serialized_example, features=self.feature_set)
        return features


# TODO(ethan): make this run in a more elagant manner with argparse values
if __name__ == "__main__":
    # TODO(ethan): get data from argparse

    dataloader = OccnetTfrecordLoader()

    sess = tf.InteractiveSession()
    # Many tf.train functions use tf.train.QueueRunner,
    # so we need to start it before we read
    # https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564
    tf.train.start_queue_runners(sess)


    # example of getting some data and using it
    # ----------------------------
    features = dataloader.get_features()

    img0 = tf.image.decode_png(features["img0"], 3).eval()
    img1 = tf.image.decode_png(features["img1"], 3).eval()

    mv0 = features["mv0"].eval()
    mvi0 = features["mvi0"].eval()
    mv1 = features["mv1"].eval()
    mvi1 = features["mvi1"].eval()

    print("mv0:\n\n{}\n\n".format(mv0))
    print("mvi0:\n\n{}\n\n".format(mvi0))
    print("mv1:\n\n{}\n\n".format(mv1))
    print("mvi1:\n\n{}\n\n".format(mvi1))

    a = {}
    a["img0"] = img0
    a["img1"] = img1

    a["mv0"] = mv0
    a["mvi0"] = mvi0
    a["mv1"] = mv1
    a["mvi1"] = mvi1

    # optionally save to a pickle file for loading in notebooks
    # with open('pickled_data.pickle', 'wb') as handle:
    #     pickle.dump(a, handle)

    # stack the images next to each other
    image = np.hstack([img0, img1])

    # save the image to visualize
    filename = "images/example_image_{}.png".format(0)
    cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))




