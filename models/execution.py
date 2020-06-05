import sys
from threading import Thread
from numba import cuda
import tensorflow as tf
import gc

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import glob

sys.path.append('../models/mrcnn')
from models.nn_models.MaskRCNN import myMaskRCNNConfig, MRCNNLogoInsertion
from models.nn_models.mrcnn import model as modellib
import cv2
from core.config import app
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"


class Compute(Thread):
    def __init__(self, request):
        Thread.__init__(self)
        self.request = request

    def run(self, config_file):
        status = process_video(config_file)
        print(status)
        self.restart_program()

    def restart_program(self):
        python = sys.executable
        os.execl(python, python, *sys.argv)


def add_audio(out_video_path):
    """
    Extract audio file from input video and add it to output video
    :param video_path: video path
    :return: output video name
    """
    video_name = out_video_path.split('/')[-1]
    audio_name = f"audio_{video_name.split('.')[0]}.mp3"
    input_video = os.path.join(app.config["UPLOAD_FOLDER"], video_name)
    output_audio = os.path.join(app.config["AUDIO_PATH"], audio_name)
    output_video = app.config["DOWNLOAD_FOLDER"] + '/sound_' + video_name
    os.system(f'ffmpeg -i {input_video} {output_audio}')
    if os.path.exists(output_audio):
        os.system(f'ffmpeg -i {out_video_path} -i {output_audio} -codec copy -shortest {output_video}')
        os.remove(out_video_path)
        os.remove(output_audio)


def process_video(config_file):

    logo_insertor = MRCNNLogoInsertion()
    logo_insertor.init_params(config_file)

    config = myMaskRCNNConfig()
    logo_insertor.model = modellib.MaskRCNN(mode="inference", config=config, model_dir='/')
    logo_insertor.model.load_weights(logo_insertor.config['model_weights_path'], by_name=True)
    source_link = logo_insertor.config['source_link']
    saving_link = logo_insertor.config['saving_link']

    logo_insertor.init_params(config_file)

    cap = cv2.VideoCapture(source_link)
    logo_insertor.fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    four_cc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(saving_link, four_cc, logo_insertor.fps, (frame_width, frame_height), True)

    while cap.isOpened():
        ret, frame = cap.read()

        if cap.get(1) % 1000 == 0:
            print(f"Inserted {round(cap.get(1)/cap.get(cv2.CAP_PROP_FRAME_COUNT) * 100, 3)}%")
            gc.collect()

        if ret:
            logo_insertor.detect_banner(frame)

            out.write(frame)
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    out.release()

    add_audio(saving_link)

    return True