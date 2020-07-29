# mobilenet_fpn6_mixnet_512_bdd_anchor_3
python3 /tf/minda/github/TKU_RVL/scripts/weight_transform.py \
--config_file '/tf/minda/github/TKU_RVL/save_models/pytorch/pytorch_model/ssdlite_mobilenet_fpn6_mixconv_512/mobilenet_fpn6_mixconv_512_bdd100k+TKU+CREDA+MOT17+taipei_E200-Copy1.yaml' \
--ckpt '/tf/minda/github/TKU_RVL/save_models/pytorch/pytorch_model/ssdlite_mobilenet_fpn6_mixconv_512/model_epoch190_better.pth' \
--check_torch "" \
--layer_name_torch '/tf/minda/github/TKU_RVL/save_models/pytorch/pytorch_model/ssdlite_mobilenet_fpn6_mixconv_512/layer_name_custom.txt' \
--layer_name_tf '/tf/minda/github/TKU_RVL/save_models/pytorch/pytorch_model/ssdlite_mobilenet_fpn6_mixconv_512/layer_name_tf.txt' \
--save_path="/tf/minda/github/TKU_RVL/out/weight_tf.pickle"