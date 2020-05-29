from models.opencv_models.OpenCVLogoInsertion import OpenCVLogoInsertion
from models.opencv_models.insert_logo_into_video import insert_logo_into_video

if __name__ == '__main__':
    banner_parameters_setting()

    input_source = 0 'SET YOUR SOURCE: 0 for image(default), 1 for video'

    if input_source == 0: # works with image
        frame_name = 'SET FRAME NAME'
        open_cv_insertion = OpenCVLogoInsertion('SET TEMPLATE NAME', frame_name, 'SET LOGO NAME')
        open_cv_insertion.build_model('SET PARAMETERS')
        cropped_frame, resized_banner_, banner_mask_cr_, switch_, f1 = open_cv_insertion.detect_banner(None)
        open_cv_insertion.insert_logo(cropped_frame, resized_banner_, banner_mask_cr_, switch_)

        # Press 'Esc' to close the resulting frame
        key = cv.waitKey(0)
        if key == 27:
            cv.destroyAllWindows()
    else: # works with video
        insert_logo_into_video('SET INPUT VIDEO NAME')
