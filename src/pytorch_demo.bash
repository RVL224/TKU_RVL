
# mobilenet_fpn_mixnetconv_bdd_better
python3 /tf/minda/github/TKU_RVL/scripts/pytorch_demo.py \
--config_file '/tf/minda/github/TKU_RVL/save_models/pytorch/pytorch_model/ssdlite_mobilenet_fpn6_mixconv_512/mobilenet_fpn6_mixconv_512_bdd100k+TKU+CREDA+MOT17+taipei_E200-Copy1.yaml' \
--ckpt '/tf/minda/github/TKU_RVL/save_models/pytorch/pytorch_model/ssdlite_mobilenet_fpn6_mixconv_512/model_epoch190_better.pth' \
--image_file '/tf/minda/github/TKU_RVL/dataset/demo/images/af4bcae9-00000000.jpg' \
--score_threshold 0.5
