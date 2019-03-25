""" server.py runs a Flask server for visualizing results from different experiments


"""
import glob
import os

from sheets import get_experiment_descriptions

from flask import Flask, render_template, send_from_directory
app = Flask(__name__)


def get_evaluations():
    """ Returns a list of lists holding the evaluation information.
    
    Returns:
        list of [ [experiment, checkpoint_name, dataset in evaluations_names], [ etc.]]
    """
    evaluations = sorted(glob.glob("../evaluations/*/"))
    evaluations_names = [os.path.basename(i[:-1]).split("##") for i in evaluations]
    return evaluations_names

@app.route('/evaluation_image/<path:path>')
def get_evaluation_image(path):
    # this takes in weird ASCII encoded characters for #
    evaluations_name = path
    return send_from_directory("../evaluations/{}/images/output".format(evaluations_name), "combined_image.png")

@app.route('/')
def hello_world():
    evaluations_names = get_evaluations()

    # add the descrption to each of the evaluations names
    experiment_descriptions = get_experiment_descriptions()

    data = []
    for experiment, checkpoint_name, dataset in evaluations_names:
        description = ""
        if experiment in experiment_descriptions:
            description = experiment_descriptions[experiment]
        data.append(
            [experiment, checkpoint_name, dataset, description]
        )

    return render_template("index.html", data=data)