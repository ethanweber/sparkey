""" executable to make sure reading a dataset works properly

this demonstrates an example of reading an occnet tfrecord

example:
python run_read_occnet_dataset.py --dataset_name 999
"""

import tensorflow as tf
import cv2
import numpy as np
import glob
import pickle
import os
import argparse

# ethan: make this import better
import sys
sys.path.append("../")
from data.occnet_data_loader import OccnetTfrecordLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="999", help='The dataset name located in the datasets folder.')
    args = parser.parse_args()

    dataset_name = args.dataset_name
    dataset_dir = os.path.join("../datasets", dataset_name)

    dataloader = OccnetTfrecordLoader(dataset_dir, occnet_data=True)

    sess = tf.Session()
    
    features = dataloader.get_features()

    for i in range(1):

        img0, img0_mask, img0_depth, img1, img1_mask, img1_depth = sess.run(
            [
                features["img0_png"][0, :, :, :3],
                features["img0_mask"],
                features["img0_depth"],
                features["img1_png"][0, :, :, :3],
                features["img1_mask"],
                features["img1_depth"]
            ]
        )

        # ethan: the output should be specified
        cv2.imwrite("run_read_occnet_dataset_{}.png".format(i), img0[...,::-1])