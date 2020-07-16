export MODEL_DIR="/home/minda/Minda/github/google_api/models/research/object_detection/custom/models/ssdlite_mobilenet_v2_coco_2018_05_09"
python3 export_saved_model_tpu.py \
--pipeline_config_file='${MODEL_DIR}/pipeline.config'\
--ckpt_path='${MODEL_DIR}/model'\
--export_dir=${MODEL_DIR}/saved_model/ \
--input_placeholder_name=image_tensor \
--input_type=image_tensor \
--use_bfloat16=True
