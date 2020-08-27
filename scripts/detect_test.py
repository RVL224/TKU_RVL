import argparse
import numpy as np
import time
import cv2
import sys

# tflite lib
import tflite_runtime.interpreter as tflite

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

def InRange(number, max, min):
  return number < max and number >= min

class detect_engine(object):
  def __init__(self, args):
    self.engine, self.input_details, self.output_details = self.init_engine(args)

    self.threshold = args.threshold

    self.labels = ["bike","bus","car","motor","person","truck","rider"]

  def init_engine(self, args):
    if(args.enable_edgetpu):
      engine = tflite.Interpreter(
                model_path=args.model_file,
                experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")])
                # num_threads=args.num_threads)
    else:
      engine = tflite.Interpreter(
                model_path=args.model_file,
                num_threads=args.num_threads)

    engine.allocate_tensors()
    input_details = engine.get_input_details()
    output_details = engine.get_output_details()

    return engine, input_details, output_details
  
  def get_input_shape(self):
    height = self.input_details[0]['shape'][1]
    width = self.input_details[0]['shape'][2]
    return width, height
  
  def get_bboxes(self, boxes, classes, scores, im_width, im_height):
    bboxes = list()
    
    for i, box in enumerate(boxes):
      if(InRange(scores[i], 1.0, self.threshold)):
        if(InRange(classes[i], len(self.labels), 1)):
          xmin, ymin, xmax, ymax = box
          bbox = {
            "bbox":{
              "xmax":int(xmax * im_width),
              "xmin":int(xmin * im_width),
              "ymax":int(ymax * im_height),
              "ymin":int(ymin * im_height)
            },
            "id": self.labels[classes[i]],
            "id_index": classes[i],
            "score": float(scores[i])
          }
          bboxes.append(bbox)
    return bboxes

  def detect(self, image, im_width, im_height):
    self.engine.set_tensor(self.input_details[0]['index'], image)
    self.engine.invoke()

    boxes = self.engine.get_tensor(self.output_details[0]['index'])
    classes = self.engine.get_tensor(self.output_details[1]['index'])
    scores = self.engine.get_tensor(self.output_details[2]['index'])

    # print(boxes)
    # print(classes)
    # print(scores)

    bboxes = self.get_bboxes(
      np.squeeze(boxes),\
      np.squeeze(classes).astype(np.int32),\
      np.squeeze(scores),\
      im_width, im_height)

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

  # window
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
    frame_expanded = np.expand_dims(frame_rgb, axis=0)
    
    tinf = time.perf_counter()
    bboxes = engine.detect(frame_expanded, im_width, im_height)
    print(time.perf_counter() - tinf, "sec")

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