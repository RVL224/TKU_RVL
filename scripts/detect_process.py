import sys
import os
import json
import argparse
import glob
import numpy as np
import time
import cv2
from PIL import Image
from tqdm import trange

""" lib """
from object_detection.utils import label_map_util
from lib.utility import load_config
from lib.utility.eval_utils import parse_gt_rec, voc_eval, voc_eval_fp_custom
from lib.utility.misc_utils import AverageMeter

DELAY_TIME = 2000

def parse_args():
  parser = argparse.ArgumentParser(description='detect process')
  parser.add_argument("--config_path",\
                      type = str,\
                      default="/tf/minda/github/TKU_RVL/cfg/demo/detect_process.json",\
                      help="config path")

  parser.add_argument("--engine",\
                      type = str,\
                      default="graph",\
                      help = "choose model (graph, tflite, tpu)")
  
  parser.add_argument("--mode",\
                      type = str,\
                      default="video",\
                      help = "choose model (video, image, images, map)")
  
  parser.add_argument("--save",\
                      type = bool,\
                      default=False,\
                      help = "save result for mode (video, image, images)")
  
  parser.add_argument("--show",\
                      type = bool,\
                      default= False,\
                      help = "show for mode (video, image, images)")
  
  args = parser.parse_args()

  return args

def mkdir(*directories):
  for directory in list(directories):
    if not os.path.exists(directory):
      os.makedirs(directory)
    else:
      pass

""" color """
ColorTable = dict({'RED': (0, 0, 255),\
                  'ORANGE': (0, 165, 255),\
                  'YELLOW': (0, 255, 255),\
                  'GREEN': (0, 255, 0),\
                  'BLUE': (255, 127, 0),\
                  'INDIGO': (255, 0, 0),\
                  'PURPLE': (255, 0, 139),\
                  'WHITE': (255, 255, 255),\
                  'BLACK': (0, 0, 0)}
)
ClassColor = dict(
        default = {'bike': ColorTable['RED'],
                   'bus': ColorTable['ORANGE'],
                   'car': ColorTable['YELLOW'],
                   'motor': ColorTable['GREEN'],
                   'person': ColorTable['WHITE'],
                   'rider': ColorTable['INDIGO'],
                   'truck': ColorTable['PURPLE'],
                  }
)

def draw_BBox(frame,bboxes,min_score_thresh = 0.2,dataset="our"):
  if(dataset == "our"):
    for bbox in bboxes:
      if(bbox['score'] >= min_score_thresh):
        cv2.rectangle(frame,\
                      (bbox['bbox']['xmin'], bbox['bbox']['ymax']),\
                      (bbox['bbox']['xmax'], bbox['bbox']['ymin']),\
                      ClassColor['default'][bbox['id']], 2)
        
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        font_scale = 1
        thickness = 1
        margin = 5
        size = cv2.getTextSize(bbox['id'], font, font_scale, thickness)
        text_width = size[0][0]
        text_height = size[0][1]
        cv2.rectangle(frame, (bbox['bbox']['xmin'], bbox['bbox']['ymax']),
                      (bbox['bbox']['xmin']+text_width, bbox['bbox']['ymax']-text_height),
                      (0, 0, 0), thickness = -1)
        
        cv2.putText(frame, bbox['id'], (bbox['bbox']['xmin'], bbox['bbox']['ymax']),
                    font, 1, ClassColor['default'][bbox['id']], 1, cv2.LINE_AA)
  else:
    for bbox in bboxes:
      if(bbox['score'] >= min_score_thresh):
        print(bbox)
        cv2.rectangle(frame,\
                      (bbox['bbox']['xmin'], bbox['bbox']['ymax']),\
                      (bbox['bbox']['xmax'], bbox['bbox']['ymin']),\
                      ColorTable['GREEN'], 2)
        
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        font_scale = 1
        thickness = 1
        margin = 5
        size = cv2.getTextSize(bbox['id'], font, font_scale, thickness)
        text_width = size[0][0]
        text_height = size[0][1]
        cv2.rectangle(frame, (bbox['bbox']['xmin'], bbox['bbox']['ymax']),
                      (bbox['bbox']['xmin']+text_width, bbox['bbox']['ymax']-text_height),
                      (0, 0, 0), thickness = -1)
        
        cv2.putText(frame, bbox['id'], (bbox['bbox']['xmin'], bbox['bbox']['ymax']),
                    font, 1, ColorTable['GREEN'], 1, cv2.LINE_AA)

def test_mAP(cfg, engine, args):
  """
    dict format : {img_id: bbox1, bbox2, ...}
    bbox format : [[xmin, ymin, xmax, ymax, id], [...], ...]

    val_preds format : [pred_box_0, pred_box_1]
    pred_box format : [image_id, x_min, y_min, x_max, y_max, score, label]
  """
  # ground true
  gt_dict, pic_dict = parse_gt_rec(cfg['VAL_MAP'], target_img_size=[512, 512], letterbox_resize=False)
  
  # predict
  val_preds = []
  for key in trange(len(gt_dict.keys())):
    frame = cv2.imread(pic_dict[key], cv2.IMREAD_COLOR)
    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    im_height, im_width, _ = frame.shape
    
    if(args.engine == 'graph'):
        frame_expanded = np.expand_dims(frame_rgb, axis=0)
    elif(args.engine == 'tflite'):
        frame_resize = cv2.resize(frame_rgb,(512,512))
        # float model
        # frame_resize = frame_resize - np.array([123, 117, 104])
        # frame_resize = frame_resize.astype(np.float32)
        
        # quantize model
        frame_resize = frame_resize.astype(np.uint8)
        frame_expanded = np.expand_dims(frame_resize, axis=0)
    elif(args.engine == 'tpu'):
        frame_rgb = cv2.resize(frame_rgb,(512,512))
        frame_expanded = Image.fromarray(frame_rgb)
    
    
    pred_content = engine.Run(frame_expanded, im_width, im_height, key)
    val_preds.extend(pred_content)
  
  print('predict success')
  
  print('mFP eval:')
  m_fp, m_det = voc_eval_fp_custom(gt_dict, val_preds, cfg['FP_THRESHOLD'])
  print("final mFP : {} \n".format(m_fp/m_det))

  rec_total, prec_total, ap_total = AverageMeter(), AverageMeter(), AverageMeter()
  
  print('mAP eval:')
  for ii in range(1, cfg['NUM_CLASSES']+1):
    npos, nd, rec, prec, ap = voc_eval(gt_dict, val_preds, ii, iou_thres=0.5, use_07_metric=cfg['USE_07_METRIC'])
    rec_total.update(rec, npos)
    prec_total.update(prec, nd)
    ap_total.update(ap, 1)
    print('Class {}: Recall: {:.4f}, Precision: {:.4f}, AP: {:.4f}'.format(ii, rec, prec, ap))
  
  mAP = ap_total.average
  print('final mAP: {:.4f}'.format(mAP))
  print("recall: {:.3f}, precision: {:.3f}".format(rec_total.average, prec_total.average))
    

def show_image(cfg,engine,args,mode = 'single'):

  if(args.show):
    # cv2.namedWindow('FRAME',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('FRAME', 600,600)
    pass

  if(mode == 'single'):
    frame = cv2.imread(cfg['SINGE_IMAGE'],cv2.IMREAD_COLOR)
    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    im_height, im_width, _ = frame.shape

    if(args.engine == 'graph'):
      frame_expanded = np.expand_dims(frame_rgb, axis=0)
    elif(args.engine == 'tflite'):
      frame_resize = cv2.resize(frame_rgb,(512,512))
      # float model
      # frame_resize = frame_resize - np.array([123, 117, 104])
      # frame_resize = frame_resize.astype(np.float32)
        
      # quantize model
      frame_resize = frame_resize.astype(np.uint8)
      frame_expanded = np.expand_dims(frame_resize, axis=0)
    elif(args.engine == 'tpu'):
      frame_rgb = cv2.resize(frame_rgb,(512,512))
      frame_expanded = Image.fromarray(frame_rgb)

    bboxes, num_detections = engine.Run(frame_expanded, im_width, im_height)
    
    draw_BBox(frame,bboxes,cfg['THRESHOLD_BBOX'],dataset=cfg['DATASET_NAME'])

    if(args.show):
      cv2.imshow("FRAME", frame)
      cv2.waitKey(DELAY_TIME)

    if(args.save):
      image_file = cfg['SINGE_IMAGE'].split('/')[-1]
      save_path = cfg['RESULT_OUT']+'/image/{}/'.format(args.engine)
      mkdir(save_path)
      cv2.imwrite(save_path+image_file, frame)

  elif(mode == 'all'):
    for image_file in os.listdir(cfg['IMAGE_DATASET']):
      frame = cv2.imread(cfg['IMAGE_DATASET']+'/'+image_file,cv2.IMREAD_COLOR)
      frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
      im_height, im_width, _ = frame.shape

      if(args.engine == 'graph'):
        frame_expanded = np.expand_dims(frame_rgb, axis=0)
      elif(args.engine == 'tflite'):
        frame_resize = cv2.resize(frame_rgb,(512,512))
        # float model
        # frame_resize = frame_resize - np.array([123, 117, 104])
        # frame_resize = frame_resize.astype(np.float32)
        
        # quantize model
        frame_resize = frame_resize.astype(np.uint8)
        frame_expanded = np.expand_dims(frame_resize, axis=0)
      elif(args.engine == 'tpu'):
        frame_rgb = cv2.resize(frame_rgb,(512,512))
        frame_expanded = Image.fromarray(frame_rgb)

      bboxes, num_detections = engine.Run(frame_expanded, im_width, im_height)

      draw_BBox(frame,bboxes,cfg['THRESHOLD_BBOX'],dataset=cfg['DATASET_NAME'])

      if(args.show):
        cv2.imshow("FRAME", frame)
        cv2.waitKey(DELAY_TIME)

      if(args.save):
        dataset = cfg['IMAGE_DATASET'].split('/')[-3]
        direction = cfg['IMAGE_DATASET'].split('/')[-1]
        save_path = cfg['RESULT_OUT']+'/image/{}/{}/{}/'.format(args.engine,dataset,direction)
        mkdir(save_path)
        cv2.imwrite(save_path+image_file, frame)

  cv2.destroyAllWindows()
  print('finish')

def show_video(cfg,engine,args):
  """ init """
  log_inference_time = []
  log_fps = []
  frame_counter = 0

  """ video """
  camera = cv2.VideoCapture(cfg['VIDEO_FILE'])
  if(args.save):
    video_file = cfg['VIDEO_FILE'].split('/')[-1]
    save_path = cfg['RESULT_OUT']+'/video/{}/'.format(args.engine)
    video_out = save_path + video_file

    mkdir(save_path)

    sz = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_out,fourcc,30, sz, True)

  if(args.show):
    # cv2.namedWindow('FRAME',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('FRAME', 640,480)
    pass

  while(camera.isOpened()):
    (grabbed, frame) = camera.read()
    if(grabbed == True):
      frame_counter += 1

      start = time.time()
      im_height, im_width, _ = frame.shape
      frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
      
      if(args.engine == 'graph'):
        frame_expanded = np.expand_dims(frame_rgb, axis=0)
      elif(args.engine == 'tflite'):
        frame_resize = cv2.resize(frame_rgb,(512,512))
        # float model
        # frame_resize = frame_resize - np.array([123, 117, 104])
        # frame_resize = frame_resize.astype(np.float32)
        
        # quantize model
        frame_resize = frame_resize.astype(np.uint8)
        frame_expanded = np.expand_dims(frame_resize, axis=0)
      elif(args.engine == 'tpu'):
        frame_rgb = cv2.resize(frame_rgb,(512,512))
        frame_expanded = Image.fromarray(frame_rgb)

      s_inference = time.time()
      bboxes, num_detections = engine.Run(frame_expanded, im_width, im_height)

      end_inference = time.time()
      inference_time = end_inference - s_inference
      print('inference time {}'.format(inference_time))

      draw_BBox(frame,bboxes,cfg['THRESHOLD_BBOX'],dataset=cfg['DATASET_NAME'])

      end = time.time()
      seconds = end - start
      fps_rate = 1 / seconds
      cv2.putText( frame, "FPS:{}".format(round(fps_rate,1)),(10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0,(255, 0, 0), 2)

      if(frame_counter > 30):
        log_inference_time.append(inference_time)
        log_fps.append(fps_rate)

      if(args.show):
        cv2.imshow("FRAME", frame)
      
      if(args.save):
        out.write(frame)

    else:
      print("no video")
      break
      
    key = cv2.waitKey(1)
    if(key==113):
        sys.exit(0)
  
  print("inference time : ",sum(log_inference_time)/len(log_inference_time))
  print("fps : ",sum(log_fps)/len(log_fps))

  camera.release()
  cv2.destroyAllWindows()
  if(args.save):
    out.release()


def main():
  args = parse_args()
  cfg = load_config.readCfg(args.config_path)

  """ label """
  label_map = label_map_util.load_labelmap(cfg["PATH_TO_LABELS"])
  categories = label_map_util.convert_label_map_to_categories(
      label_map, max_num_classes=cfg['NUM_CLASSES'], use_display_name=True)
  category_index = label_map_util.create_category_index(categories)
  
  if(args.engine == 'graph'):
    from lib.engine.float_engine import Engine
    engine = Engine(cfg,category_index)
  elif(args.engine == 'tflite'):
    from lib.engine.tflite_engine import Engine
    engine = Engine(cfg,category_index)
  elif(args.engine == 'tpu'):
    from lib.engine.tpu_engine import Engine
    engine = Engine(cfg,category_index)
  
  if(args.mode == 'video'):
    show_video(cfg,engine,args)
  elif(args.mode == 'image'):
    show_image(cfg,engine,args,'single')
  elif(args.mode == 'images'):
    show_image(cfg,engine,args,'all')
  elif(args.mode == 'map'):
    test_mAP(cfg,engine,args)

if __name__ == "__main__":
  main()
