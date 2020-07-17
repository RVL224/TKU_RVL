#!/bin/bash

# model param
CONFIG_FILE="/tf/minda/github/TKU_RVL/cfg/train/ssdlite_mobilenet_v2_fpn6_512_mixconv_anchor_3_bdd_better.config"

# pytorch model
WEIGHT_FILE="/tf/minda/github/TKU_RVL/save_models/pytorch/pytorch_model/ssdlite_mobilenet_fpn6_mixconv_512/weight_tf.pickle"
LAYERS_FILE="/tf/minda/github/TKU_RVL/save_models/pytorch/pytorch_model/ssdlite_mobilenet_fpn6_mixconv_512/layer_name_custom.txt"

# output
MODEL_DIR="/tf/minda/github/TKU_RVL/out/ssdlite_mobilenet_fpn6_mixconv_512_bdd"

CUDA_VISIBLE_DEVICES=0 python3 /tf/minda/github/TKU_RVL/models/research/object_detection/legacy/train.py \
--logtostderr \
--pipeline_config_path=${CONFIG_FILE} \
--pytorch_weight_path=${WEIGHT_FILE} \
--pytorch_layers_path=${LAYERS_FILE} \
--load_pytorch=False \
--train_dir=${MODEL_DIR}