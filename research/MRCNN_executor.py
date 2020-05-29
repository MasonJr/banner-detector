import sys
import time
sys.path.append('models/nn_models/')
from models.nn_models.MaskRCNN import MRCNNLogoInsertion, myMaskRCNNConfig
from models.nn_models.mrcnn import model as modellib
import cv2
import os




if __name__ == '__main__':

    start = time.time()

    logo_insertor = MRCNNLogoInsertion()
    logo_insertor.init_params("models/configurations/model_parameters.yaml")

    config = myMaskRCNNConfig()
    logo_insertor.model = modellib.MaskRCNN(mode="inference", config=config, model_dir='./')
    logo_insertor.model.load_weights(logo_insertor.config['model_weights_path'], by_name=True)
    # load parameters
    source_type = logo_insertor.config['source_type']
    source_link = logo_insertor.config['source_link']
    save_result = logo_insertor.config['save_result']
    saving_link = logo_insertor.config['saving_link']

    print('start')

    if source_type == 0:
        print("Detection step")
        cap = cv2.VideoCapture(source_link)
        logo_insertor.fps = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, frame = cap.read()

            if cap.get(1) % 1000 == 0:
                print(f"Still need to process {cap.get(cv2.CAP_PROP_FRAME_COUNT) - cap.get(1)} frames")

            if ret:
                logo_insertor.detect_banner(frame)
            else:
                break

        cap.release()

        print('Insertion step')

        logo_insertor.frame_num = 0
        logo_insertor.before_smoothing = False
        logo_insertor.init_params("models/configurations/model_parameters.yaml")

        cap = cv2.VideoCapture(source_link)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        four_cc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter(saving_link, four_cc, logo_insertor.fps, (frame_width, frame_height), True)

        while cap.isOpened():
            ret, frame = cap.read()

            if cap.get(1) % 1000 == 0:
                print(f"Still need to process {cap.get(cv2.CAP_PROP_FRAME_COUNT) - cap.get(1)} frames")

            if ret:
                logo_insertor.detect_banner(frame)
                logo_insertor.insert_logo()

                if save_result:
                    out.write(frame)
            else:
                break

        cap.release()
        cv2.destroyAllWindows()
        out.release()
        timing = time.time() - start
        print(f"The processing video took {timing//60} minutes {round(timing%60)} seconds")

    else:
        frame = cv2.imread(source_link, cv2.IMREAD_UNCHANGED)

        logo_insertor.detect_banner(frame)

        logo_insertor.frame_num = 0
        logo_insertor.before_smoothing = False

        logo_insertor.detect_banner(frame)
        logo_insertor.insert_logo()

        if save_result:
            cv2.imwrite(saving_link, frame)
        cv2.imshow('Image (press Q to close)', frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
