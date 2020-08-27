#!/bin/bash

# tflite
# python3 /tf/minda/github/TKU_RVL/scripts//detect_test.py \
# -v /home/iclab/minda/detect_demo/video/taipei_road2.mp4 \
# -m /home/iclab/minda/dl_models/mobilenet_30.tflite \
# -th 0.5 \
# --num_threads 1

# edgetpu
python3 /tf/minda/github/TKU_RVL/scripts/detect_test.py \
-v /home/iclab/minda/detect_demo/video/taipei_road2.mp4 \
-m /home/iclab/minda/dl_models/mobilenet_30_edgetpu.tflite \
-th 0.5 \
--num_threads 1 \
--enable_edgetpu

# python3 /tf/minda/github/TKU_RVL/scripts/detect_edgetpu_test.py \
# -v /home/iclab/minda/detect_demo/video/taipei_road2.mp4 \
# -m /home/iclab/minda/dl_models/mobilenet_30_edgetpu.tflite \
# -th 0.5 \
# --num_threads 1 \
# --enable_edgetpu