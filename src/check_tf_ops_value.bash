#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 /tf/minda/github/TKU_RVL/scripts/check_tf_ops_value.py \
--model_path '/tf/minda/github/TKU_RVL/save_models/tensorflow/tensorflow_model/ssdlite_mobilenet_fpn6_mixconv_512_bdd/save_model/frozen_inference_graph.pb' \
--image_file '/tf/minda/github/TKU_RVL/dataset/demo/images/af4bcae9-00000000.jpg'
