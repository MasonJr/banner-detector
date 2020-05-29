sudo docker run --gpus 1 -it --rm -p 5089:5089 --name bb \
--mount type=bind,source=$(pwd)/instance/logotypes,target=/usr/src/app/instance/logotypes \
--mount type=bind,source=$(pwd)/instance/upload/,target=/usr/src/app/instance/upload \
--mount type=bind,source=$(pwd)/instance/download/,target=/usr/src/app/instance/download \
--mount type=bind,source=$(pwd)/instance/weights/,target=/usr/src/app/instance/weights \
--mount type=bind,source=$(pwd)/instance/audio/,target=/usr/src/app/instance/audio \
--mount type=bind,source=$(pwd)/models/configurations/,target=/usr/src/app/models/configurations \
--mount type=bind,source=$(pwd)/models/frame_mask,target=/usr/src/app/models/frame_mask \
banner-detector:1.0
