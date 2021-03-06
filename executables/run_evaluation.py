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

specify a checkpoint:
python run_evaluation.py --experiment_name 19-03-24_02:33:20 --dataset_name 000 --checkpoint model.ckpt-9576

to run with keypointnet
add --keypointnet
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
    parser.add_argument('--checkpoint', type=str, default=None, help='The checkpoint to use to running evaluation.')
    parser.add_argument('--keypointnet', default=False, action='store_true', help='Use this flag if running on keypointnet data.')
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

    # figure out checkpoint_name, which will be used in the folder name
    if args.checkpoint is None:
        checkpoint_path = os.path.join(experiment_path, "checkpoint")
        with open(checkpoint_path) as f:
            first_line = f.readline()
        start_index = first_line.find("\"")
        end_index = first_line.find("\"", start_index+1)
        checkpoint_name = first_line[start_index+1:end_index]
    else:
        checkpoint_name = args.checkpoint

    # create the evaluation folder
    evaluation_name = "{}##{}##{}".format(experiment_name, checkpoint_name, dataset_name)
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
    dataloader = OccnetTfrecordLoader(dataset_dir, occnet_data=not args.keypointnet)
    sess = tf.Session()
    features = dataloader.get_features()

    # ethan: make the number of images a parameter
    for i in range(20):

        img0, img1 = sess.run(
            [
                features["img0_png"][0, :, :, :3],
                features["img1_png"][0, :, :, :3]
            ]
        )

        # ethan: the output should be specified
        current_image_file_left = os.path.join(images_folder, "{:05d}_left.png".format(i))
        current_image_file_right = os.path.join(images_folder, "{:05d}_right.png".format(i))
        cv2.imwrite(current_image_file_left, img0[...,::-1])
        cv2.imwrite(current_image_file_right, img1[...,::-1])

    # now that the images are written, we can run inference on the images

    main_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../network/main.py")
        
    command_to_run = "python {} --model_dir={} --input={} --predict".format(
        main_file_path, 
        experiment_path,
        images_folder)

    if args.checkpoint:
        command_to_run += " --latest_filename {}".format(args.checkpoint)

    if args.keypointnet:
        command_to_run += " --keypointnet"

    # close the session before running the prediction command
    sess.close()

    print("\nRunning command:\n {}\n\n".format(command_to_run))
    os.system(command_to_run)

    # now go through predict folder and create a full image from the predictions
    output_images_glob = os.path.join(images_folder, "output/*.png")
    image_files = sorted(glob.glob(output_images_glob))
    images = []
    count = 0
    for i in range(10):
        row = []
        for j in range(2):
            image = cv2.imread(image_files[count])
            row.append(image)
            count += 1
        images.append(row)
    rows = [np.hstack(row) for row in images]
    full_image = np.vstack(rows)
    cv2.imwrite(os.path.join(images_folder, "output/combined_image.png"), full_image)