import argparse
import numpy as np
import time
import cv2
import sys
from PIL import Image

# edgetpu lib
from edgetpu.detection.engine import DetectionEngine


def parse_args():
  parser = argparse.ArgumentParser(description='detect demo')

  parser.add_argument(
    "-v", "--video",
    default="/home/iclab/minda/detect_demo/video/taipei_road2.mp4", \
    help="video to be detected")
  
  parser.add_argument(
    "-m", "--model_file", \
    default="/home/iclab/minda/dl_models/mobilenet_30_edgetpu.tflite", \
    help=".tflite model to be executed")
  
  parser.add_argument(
    "-th","--threshold",
    type=float,
    default=0.5,
    help="detect threshold")
  
  parser.add_argument(
    "--enable_edgetpu", \
    action="store_true", \
    help=".tflite model to be executed")

  parser.add_argument(
    "--num_threads",
    type=int,
    default=1,
    help="number of threads")
  
  args = parser.parse_args()
  
  return args

class detect_engine(object):
  def __init__(self, args):
    self.engine = self.init_engine(args)

    self.threshold = args.threshold

    self.labels = ["bike","bus","car","motor","person","truck","rider"]

  def init_engine(self, args):
    
    engine = DetectionEngine(args.model_file)

    return engine
  
  def get_input_shape(self):
    _, height, width, channel = self.engine.get_input_tensor_shape()
    return width, height
  
  def get_bboxes(self, ans, im_width, im_height):
    bboxes = list()
    
    for obj in ans:
      box = obj.bounding_box

      bbox = {
        "bbox":{
          "xmax":int(box[1][1] * im_width),
          "xmin":int(box[0][1] * im_width),
          "ymax":int(box[1][0] * im_height),
          "ymin":int(box[0][0] * im_height)
        },
        "id": self.labels[obj.label_id],
        "id_index": obj.label_id,
        "score": float(obj.score)
      }
      bboxes.append(bbox)
    return bboxes

  def detect(self, image, im_width, im_height):
    ans = self.engine.detect_with_image(image,\
                                      threshold=self.threshold,\
                                      keep_aspect_ratio=False,\
                                      relative_coord=True,\
                                      top_k=10)
    
    print(self.engine.get_inference_time())

    bboxes = self.get_bboxes(ans, im_width, im_height)

    return bboxes

def draw(frame, bboxes):
  for bbox in bboxes:
    cv2.rectangle(frame,\
                  (bbox['bbox']['xmin'],bbox['bbox']['ymax']),\
                  (bbox['bbox']['xmax'],bbox['bbox']['ymin']),\
                  (0,255,0),2)

def run(args):
  # init detect engine
  engine = detect_engine(args)
  m_width, m_height = engine.get_input_shape()
   
  # show window setting
  cv2.namedWindow("output", cv2.WINDOW_NORMAL)
  cv2.resizeWindow("output", 640, 480)

  camera = cv2.VideoCapture(args.video)
  while(camera.isOpened()):

    (grabbed, frame) = camera.read()
    if(grabbed == False):
      break
    
    # run innference
    im_height, im_width, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (m_width, m_height))
    frame_expanded = Image.fromarray(frame_rgb)

    tinf = time.perf_counter()
    bboxes = engine.detect(frame_expanded, im_width, im_height)
    # print(time.perf_counter() - tinf, "sec")
    
    # draw bounding box
    draw(frame, bboxes)
    cv2.imshow("output",frame)

    key = cv2.waitKey(1)
    if(key == 113):
      break
  
  camera.release()
  cv2.destoryAllWindows()

def main():
  args = parse_args()
  
  run(args)

if __name__ == "__main__":
  main()