import os
import sys

from flask import Flask

sys.path.append('../')

app = Flask("banner")
app.url_map.strict_slashes = False

instance = 'instance'
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
instance_path = os.path.join(project_path, instance)

app.config["UPLOAD_FOLDER"] = instance_path + '/upload'
app.config["DOWNLOAD_FOLDER"] = instance_path + '/download'
app.config["WEIGHT_FOLDER"] = instance_path + '/weights'
app.config["LOGO_FOLDER"] = instance_path + '/logotypes'
app.config["AUDIO_PATH"] = instance_path + '/audio'

app.config["MODEL_FOLDER"] = project_path + '/models'
app.config["CONFIG_PATH"] = app.config["MODEL_FOLDER"] + '/configurations/model_parameters.yaml'
app.config["MASK_PATH"] = app.config["MODEL_FOLDER"] + '/frame_mask'

if not os.path.exists(instance_path):
    os.mkdir(instance_path)

if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.mkdir(app.config["UPLOAD_FOLDER"])

if not os.path.exists(app.config["DOWNLOAD_FOLDER"]):
    os.mkdir(app.config["DOWNLOAD_FOLDER"])

if not os.path.exists(app.config["WEIGHT_FOLDER"]):
    os.mkdir(app.config["WEIGHT_FOLDER"])

if not os.path.exists(app.config["LOGO_FOLDER"]):
    os.mkdir(app.config["LOGO_FOLDER"])

if not os.path.exists(app.config["AUDIO_PATH"]):
    os.mkdir(app.config["AUDIO_PATH"])

if not os.path.exists(app.config["MODEL_FOLDER"]):
    os.mkdir(app.config["MODEL_FOLDER"])

if not os.path.exists(app.config["MASK_PATH"]):
    os.mkdir(app.config["MASK_PATH"])
