""" executable that runs an evaluation for a given experiment


- creates a directory for output
- write pngs
- computes results on these pngs

args:
 - experiment_name:
 - dataset_name:

output:
outputs to the evaluations/ folder. creates a folder with name of experiment and dataset evaluated on

example:
python run_evaluation.py --experiment_name 19-03-23_10:09:07 --dataset_name 999
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
    parser.add_argument('--experiment_name', type=str, default=None, help='The experiment name located in the experiments folder.')
    parser.add_argument('--dataset_name', type=str, default=None, help='The dataset name located in the datasets folder.')
    args = parser.parse_args()

    experiment_name = args.experiment_name
    dataset_name = args.dataset_name

    if experiment_name is None or dataset_name is None:
        raise ValueError("One or both of experiment_name and dataset_name are None")

    # update the names with the full paths
    experiment_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../experiments",
        experiment_name)
    dataset_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../datasets",
        dataset_name)

    # create the evaluation folder
    evaluation_name = "{}##{}".format(dataset_name, experiment_name)
    evaluation_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../evaluations",
        evaluation_name)
    if not os.path.exists(evaluation_folder):
        os.makedirs(evaluation_folder)

    # make image folder in the evaluation folder
    images_folder = os.path.join(evaluation_folder, "images")
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)


    # add images to the folder
    dataset_dir = dataset_path
    dataloader = OccnetTfrecordLoader(dataset_dir, occnet_data=True)
    sess = tf.Session()
    features = dataloader.get_features()

    # ethan: make the number of images a parameter
    for i in range(20):

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
        current_image_file = os.path.join(images_folder, "{:05d}.png".format(i))
        cv2.imwrite(current_image_file, img0[...,::-1])

    # now that the images are written, we can run inference on the images

    main_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../network/main.py")
    command_to_run = "python {} --model_dir={} --input={} --predict".format(
        main_file_path, 
        experiment_path,
        images_folder)

    # close the session before running the prediction command
    sess.close()

    print("\nRunning command:\n {}\n\n".format(command_to_run))
    os.system(command_to_run)