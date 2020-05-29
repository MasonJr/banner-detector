from models.nn_models.UnetLogoInsertion import UnetLogoInsertion

if __name__ == '__main__':

    logo_insertor = UnetLogoInsertion()
    logo_insertor.build_model('models/configurations/model_parameters_setting')

    # load parameters
    source_type = logo_insertor.model_parameters['source_type']
    source_link = logo_insertor.model_parameters['source_link']
    save_result = logo_insertor.model_parameters['save_result']
    saving_link = logo_insertor.model_parameters['saving_link']

    # works with video
    if source_type == 0:

        # preprocessing (detection and smoothing points)
        cap = cv2.VideoCapture(source_link)
        while (cap.isOpened()):
            ret, frame = cap.read()

            if ret:
                logo_insertor.detect_banner(frame)
                print(logo_insertor.frame_num)
            else:
                break
        cap.release()

        logo_insertor.frame_num = 0
        logo_insertor.before_smoothing = False

        # logo insertion
        cap = cv2.VideoCapture(source_link)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        four_cc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(saving_link, four_cc, 30, (frame_width, frame_height), True)

        while (cap.isOpened()):
            ret, frame = cap.read()

            if ret:
                logo_insertor.detect_banner(frame)
                logo_insertor.insert_logo()

                if save_result:
                    out.write(frame)

                cv2.imshow('Video (press Q to close)', frame)
                if cv2.waitKey(23) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()
        out.release()

    # works with image
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
