#!/bin/bash

# output
MODEL_DIR="/tf/minda/github/TKU_RVL/out/ssdlite_mobilenet_fpn6_mixconv_512_bdd"
CKPT_STEP="0"

# CUDA_VISIBLE_DEVICES=1
# creates the frozen inference graph in fine_tune_model for test
CUDA_VISIBLE_DEVICES=0 python3 /tf/minda/github/TKU_RVL/models/research/object_detection/export_inference_graph.py \
--input_type=image_tensor \
--pipeline_config_path=${MODEL_DIR}/pipeline.config \
--trained_checkpoint_prefix=${MODEL_DIR}/model.ckpt-${CKPT_STEP} \
--output_directory=${MODEL_DIR}/save_model


# cal flops and params
# CUDA_VISIBLE_DEVICES=0 python3 /tf/minda/github/TKU_RVL/models/research/object_detection/export_inference_graph.py \
# --input_type=image_tensor \
# --input_shape=1,512,512,3 \
# --pipeline_config_path=${MODEL_DIR}/pipeline.config \
# --trained_checkpoint_prefix=${MODEL_DIR}/model.ckpt-${CKPT_STEP} \
# --output_directory=${MODEL_DIR}/save_model