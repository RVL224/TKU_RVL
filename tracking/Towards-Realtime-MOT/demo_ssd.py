import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm

from tracker.multitracker import JDETracker, ssd_JDETracker
from utils import visualization as vis
from utils.utils import *
from utils.io import read_results
from utils.log import logger
from utils.timer import Timer
from utils.evaluation import Evaluator
import utils.datasets as datasets
import torch
from ssd_track import ssd_eval_seq

from ssd.config import cfg

def track(opt, cfg):
    logger.setLevel(logging.INFO)
    result_root = opt.output_root if opt.output_root!='' else '.'
    mkdir_if_missing(result_root)

    # run tracking
    timer = Timer()
    accs = []
    n_frame = 0

    logger.info('start tracking...')
    dataloader = datasets.ssd_LoadVideo(opt.input_video, opt.img_size, cfg)
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate 

    frame_dir = None if opt.output_format=='text' else osp.join(result_root, 'frame')

    _, average_time, _= ssd_eval_seq(opt, dataloader, 'mot', result_filename,
                 save_dir=frame_dir, show_image=False, frame_rate=frame_rate, cfg=cfg)
    print('fps = %f'%(1/average_time))

    if opt.output_format == 'video':
        output_video_path = osp.join(result_root, 'result.mp4')
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(osp.join(result_root, 'frame'), output_video_path)
        os.system(cmd_str)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='demo.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    #parser.add_argument('--weights', type=str, default='weights/vgg.pt', help='path to weights file')
    #parser.add_argument('--weights', type=str, default='weights/m2det_e30_b32_1e-3_5e-4_Adam.pt', help='path to weights file')
    #parser.add_argument('--weights', type=str, default='weights/mobilenet_fpn_test_512_b32_e100_1e-3_Adam_1emb_gpu0.pt', help='path to weights file')
    #parser.add_argument('--weights', type=str, default='weights/mobilyenet_fpn_test_512_b32_e130_1e-3_adam_1emb_embmask5.pt', help='path to weights file') 
    
    #parser.add_argument('--weights', type=str, default='weights/mobilenet_fpn7_cfe_512_b32_e100_1e-3_Adam_1emb_lossmeanmask_gpu0.pt', help='path to weights file')
    parser.add_argument('--weights', type=str, default='weights/epoch_119.pt', help='path to weights file')


    parser.add_argument('--img-size', type=int, default=(512, 512), help='size of each image dimension')
    #parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')
    parser.add_argument('--min-box-area', type=float, default=200, help='filter out tiny boxes')
    parser.add_argument('--track-buffer', type=int, default=30, help='tracking buffer')
    #parser.add_argument('--input-video', type=str, default='video/half.mp4',help='path to the input video')
    parser.add_argument('--input-video', type=str, default='video/15s_demo.mp4',help='path to the input video')

    parser.add_argument('--output-format', type=str, default='video', help='expected output format, can be video, or text')
    parser.add_argument('--output-root', type=str, default='video/result', help='expected output root path')
    opt = parser.parse_args()

    ssd_cfg_path = '/home/eervl224/Towards-Realtime-MOT/mobilenet_fpn_test_512_bdd100k_E200.yaml'
    #ssd_cfg_path = '/home/eervl224/Towards-Realtime-MOT/vgg_ssd512_voc0712.yaml'
    #ssd_cfg_path = '/home/eervl224/Towards-Realtime-MOT/vgg_m2det_512.yaml'
    #ssd_cfg_path = '/home/eervl224/Towards-Realtime-MOT/mobilenet_fpn7_cfe_512_voc_E200.yaml'

    cfg.merge_from_file(ssd_cfg_path)
    cfg.freeze()

    print(opt, end='\n\n')

    track(opt , cfg)

