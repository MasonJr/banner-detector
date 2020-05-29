# LOGO INSERTION

## There are three different mechanisms for inserting a logo.

### The first is based on the use of classical methods of computer vision.

The algorithm's job is to find the desired banner ad for a given template and replace it with the banner one you want.


### The second is based on the use of a neural network from the architect U-Net.

This method performs the same function as the previous one, but in a different way. Finding the right banner performs the neural network, also different method of insertion of the logo.

### The latest, most accurate method is to use the Mask-RCNN architecture for better banner detection and class-matching banners.

The operation of this method is very similar to the second type of mechanism, but the complexity of the algorithm is much greater, which gives more opportunities to choose banners for replacement.


#### The general algorithm of the mechanisms is as follows:
- We find a banner in 1 case using key points, otherwise using a neural network.
- The first method requires pre-processing to detect the shape by color. The other two methods are a set of masks for each banner, where you also need to find a figure.
- Prepare a logo for insertion, namely color alignment and transformation.
- Insert the logo into the detected field replacing the detected figure.


The current version allows you to select 8 types of banner for replacement and set time intervals for banner insertion. In the near future, a mechanism for inserting cascade type banners will be implemented.


### To run the mechanism you need to:
- Download the repository with all consisting files;

- Add video that will change, banner replacement templates, model weights for the last two methods;

- **Please take into account if you want to use the first method:** the best way to create a template is to find the frame in the video where the required field is clearly visible. To do so use function frames_capture from insert_logo_into_video.py to get all frames from the required video (remove the comment at function call in line 98, set video file name, create folder to paste frames and set folder path in line 36) and choose required frame to create the template;

- Install or upgrade necessary packages from requirements.txt;

- For additional settings, open the file ``` configurations/model_parameters.yaml ``` **(hereafter the configuration file)** in ```models``` folder and configure pipeline for yourself;

- Set path for **input/output video and model weight** in configuration file at coresponing fields;

- If you want to insert logotype in unique frame, you should set up ```source_type``` in configuration file as **1** else **0**;

- To select the type of banner to replace (only relevant for the third mechanism), you must set in the ```replace``` field in the configuration file the indexes of the banner classes and the corresponding paths to the files to be inserted. For two other methods just set path for logo in the field ```logo_link```;

- Depends on the method what you want to use, select the executor, for example:
  * OpenCV Model - ```OpenCVModel_executor.py```
  * MascRCNN Model - ```MRCNN_executor.py```
  * UNet Model - ```UnetModel_executor.py```

   and run with comand ```python <model>_executor.py```.
