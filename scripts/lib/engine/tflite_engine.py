# from tflite_runtime.interpreter import Interpreter
# from tflite_runtime.interpreter import load_delegate
import tensorflow as tf
import numpy as np

def InRange(number, max, min):
  return number < max and number >= min

class Engine(object):
  def __init__(self,cfg,category_index):
    self.cfg = cfg
    self.category_index = category_index
    self.engine, self.input_details, self.output_details = self.Init_Engine()
    
    self.val_threshold = cfg['VAL_THRESHOLD']

  def Init_Engine(self):
    print(self.cfg['PATH_TFLITE'])
#     engine = Interpreter(self.cfg['PATH_TFLITE'])
    engine = tf.lite.Interpreter(self.cfg['PATH_TFLITE'])
    engine.allocate_tensors()

    input_details = engine.get_input_details()
    output_details = engine.get_output_details()

    print(input_details)

    return engine, input_details, output_details
  
  def Get_Results(self,boxes, classes, scores, im_width, im_height,min_score_thresh=.2):
    bboxes = list()
    for i, box in enumerate(boxes):
      if(InRange(scores[i], 1.0, min_score_thresh)):
        if(InRange(classes[i], len(self.category_index), 1)):
          # ymin, xmin, ymax, xmax = box
          xmin, ymin, xmax, ymax = box
          bbox = {
            'bbox': {
                'xmax': int(xmax * im_width),
                'xmin': int(xmin * im_width),
                'ymax': int(ymax * im_height),
                'ymin': int(ymin * im_height)
            },
            'id': self.category_index[classes[i]+1]['name'],
            'id_index': classes[i]+1,
            'score': float(scores[i])
          }
          bboxes.append(bbox)
    return bboxes
  
  def Run(self, image, im_width, im_height, image_id=None):
    self.engine.set_tensor(self.input_details[0]['index'], image)
    self.engine.invoke()

    boxes = self.engine.get_tensor(self.output_details[0]['index'])
    classes = self.engine.get_tensor(self.output_details[1]['index'])
    scores = self.engine.get_tensor(self.output_details[2]['index'])
    num_detections = self.engine.get_tensor(self.output_details[3]['index'])

    # print(len(boxes))
    # print(boxes)
    # print(classes)
    # print(scores)
    # print(num_detections)    
    
    if(image_id == None):
      bboxes = self.Get_Results(np.squeeze(boxes),\
                              np.squeeze(classes).astype(np.int32),\
                              np.squeeze(scores),\
                              im_width, im_height)
      return bboxes, num_detections
    else:
      bboxes = self.Get_Results(
                 np.squeeze(boxes),\
                 np.squeeze(classes).astype(np.int32),\
                 np.squeeze(scores),\
                 im_width, im_height,\
                 self.val_threshold)  
        
      pred_content = []
      for item in bboxes:
        pred_content.append(
        [image_id, \
         item['bbox']['xmin'], \
         item['bbox']['ymin'], \
         item['bbox']['xmax'], \
         item['bbox']['ymax'], \
         item['score'], \
         item['id_index']])
        
      return pred_content
  

if __name__ == "__main__":
    main()
