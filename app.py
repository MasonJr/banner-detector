import sys
from collections import defaultdict

from werkzeug.utils import secure_filename

sys.path.append('models/')
sys.path.append('models/nn_models/')

import yaml
import os
from core.config import app

from models.execution import Compute
from flask import request, render_template, redirect
from core.tools import convert_time


@app.before_request
def before_request():

    with open(app.config["CONFIG_PATH"], 'r') as file:
        model_parameters = yaml.load(file, Loader=yaml.FullLoader)

    model_parameters["mask_path"] = app.config["MASK_PATH"]
    model_parameters["model_weights_path"] = app.config["WEIGHT_FOLDER"] + '/mrcnn.h5'

    with open(app.config["CONFIG_PATH"], 'w') as write_file:
        documents = yaml.dump(model_parameters, write_file)


@app.route('/')
def init():
    return redirect('/set_video')

@app.route('/set_video', methods=["POST", "GET"])
def get_video_path():

    if request.method == 'GET':
        return render_template("video.html")
    else:
        video_path = request.form['video_path']

        with open(app.config["CONFIG_PATH"], 'r') as params_file:
            model_parameters = yaml.load(params_file, Loader=yaml.FullLoader)

        video_name = video_path.split('/')[-1]

        model_parameters['source_link'] = os.path.join(app.config["UPLOAD_FOLDER"], video_name)
        model_parameters['saving_link'] = os.path.join(app.config["DOWNLOAD_FOLDER"], video_name)
        with open(app.config["CONFIG_PATH"], 'w') as write_file:
            documents = yaml.dump(model_parameters, write_file)

        return redirect('/periods')


@app.route('/periods', methods=["POST", "GET"])
def set_time_periods():
    if request.method == "GET":
        return render_template('periods.html')
    else:

        data = request.form

        periods = defaultdict(dict)

        for timepoint in data:
            value = data[timepoint]
            point, n = timepoint.split('_')
            period = f"period_{n}"
            if value:
                periods[period][point] = value
            else:
                if period in periods:
                    del periods[period]

        periods = dict(periods)

        with open(app.config["CONFIG_PATH"], 'r') as file:
            model_parameters = yaml.load(file, Loader=yaml.FullLoader)

        model_parameters['periods'] = {}
        if periods:
            for period in periods:
                model_parameters['periods'][period] = {}
                start, finish = periods[period].values()
                model_parameters['periods'][period]['start'] = convert_time(start)
                model_parameters['periods'][period]['finish'] = convert_time(finish)

        with open(app.config["CONFIG_PATH"], 'w') as write_file:
            documents = yaml.dump(model_parameters, write_file)

        return redirect('/process')


@app.route('/process', methods=["POST", "GET"])
def process():

    if request.method == "GET":
        return render_template('process.html')
    else:
        thread_a = Compute(request.__copy__)
        thread_a.run(app.config["CONFIG_PATH"])

        return render_template("finished.html")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port="5089", threaded=False)
