import glob
import os

from flask import Flask, render_template, send_from_directory
app = Flask(__name__)


def get_evaluations():
    evaluations = sorted(glob.glob("../evaluations/*/"))
    evaluations_names = [os.path.basename(i[:-1]).split("##") for i in evaluations]
    # print(evaluations)
    # print(evaluations_names)

    return evaluations_names


def get_image_from_evaluation_name():
    pass

@app.route('/evaluation_image/<path:path>')
def get_evaluation_image(path):
    evaluations_name = path
    # print(evaluations_name)
    # evaluations_name = "19-03-24_01:05:24##model.ckpt-69167##000"
    return send_from_directory("../evaluations/{}/images/output".format(evaluations_name), "00000_left.png")

@app.route('/')
def hello_world():
    evaluations_names = get_evaluations()
    # return 'Hello, World!'
    return render_template("index.html", evaluations_names=evaluations_names)