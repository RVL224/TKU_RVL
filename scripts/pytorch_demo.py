import os
import sys
import argparse
import numpy as np
import cv2

import torch

# library
from ssd.config import cfg
from ssd.utils.checkpoint import CheckPointer
from ssd.modeling.detector import build_detection_model
from ssd.data.transforms import build_transforms
from ssd.modeling.box_head.inference import PostProcessor

def parse_args():
  parser = argparse.ArgumentParser(description="SSD Demo.")
  parser.add_argument(
    "--config-file",
    default="",
    metavar="FILE",
    help="path to config file",
    type=str,
  )
  parser.add_argument("--ckpt", type=str, default=None, help="Trained weights.")
  parser.add_argument("--score_threshold", type=float, default=0.7)
  parser.add_argument("--image_file", default=None, type=str, help='Specify a video dir to do prediction.')
  parser.add_argument("--dataset_type", default="voc", type=str, help='Specify dataset type. Currently support voc and coco.')
    
  parser.add_argument(
    "opts",
    help="Modify config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER,
  )

  return parser.parse_args()

def check_tensor_value(tensor, mode='tf'):
  if(mode == 'tf'):
    tensor_np = np.transpose(tensor.detach().numpy(),(0,2,3,1))
  else:
    tensor_np = tensor.detach().numpy()
    
  print("tensor shape: {}".format(tensor_np.shape))
  print("tensor value: ")
  print(tensor_np)

  sys.exit()

@torch.no_grad()
def run_demo_image(cfg, ckpt, score_threshold, image_name, dataset_type):
  class_names = ('__background__',
                 'bike', 'bus', 'car','motor','person','truck','rider')

  # init device
  device = torch.device(cfg.MODEL.DEVICE)
  cpu_device = torch.device("cpu")

  # init model 
  model = build_detection_model(cfg)
  model = model.to(device)
  checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
  checkpointer.load(ckpt, use_latest=ckpt is None)
  weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
  print('Loaded weights from {}'.format(weight_file))
  
  # build preporcess
  transforms = build_transforms(cfg, is_train=False)
  
  model.eval()
  
  if(image_name is not None):
    frame = cv2.imread(image_name,cv2.IMREAD_COLOR)
    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
    # preporcess
    width, height = frame.shape[0:2]
    images = transforms(frame_rgb)[0].unsqueeze(0)
    
    # check_tensor_value(images)

def main():
  args = parse_args()
  print(args)
    
  cfg.merge_from_file(args.config_file)
  cfg.merge_from_list(args.opts)
  cfg.freeze()
  
  print("Loaded configuration file {}".format(args.config_file))
  with open(args.config_file, "r") as cf:
    config_str = "\n" + cf.read()
    print(config_str)
  print("Running with config:\n{}".format(cfg))

  run_demo_image(
    cfg=cfg,
    ckpt=args.ckpt,
    score_threshold=args.score_threshold,
    image_name=args.image_file,
    dataset_type=args.dataset_type)

if __name__ == '__main__':
  main()