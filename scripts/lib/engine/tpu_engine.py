from edgetpu.detection.engine import DetectionEngine
from edgetpu.basic.basic_engine import BasicEngine
import numpy as np

class Engine(object):
  def __init__(self,cfg,category_index):
    self.cfg = cfg
    self.category_index = category_index
    self.engine = self.Init_Engine()
    
    self.val_threshold = cfg['VAL_THRESHOLD']

    self.max_num = 100
  
  def Init_Engine(self):
    engine = DetectionEngine(self.cfg['PATH_TPU'])

    return engine
  
  def Get_Results(self,out, im_width, im_height, min_score_thresh=.2):
    bboxes = list()
    for item in out:
      if item.score > min_score_thresh:
        box = item.bounding_box
        
        # default
        # bbox = {
        #   'bbox': {
        #       'xmax': int(box[1][0]),
        #       'xmin': int(box[0][0]),
        #       'ymax': int(box[1][1]),
        #       'ymin': int(box[0][1])
        #   },
        #   'id': self.category_index[item.label_id+1]['name'],
        #   'score': float(item.score)
        # }
        
        # for pytorch
        bbox = {
          'bbox': {
              'xmax': int(box[1][1]* im_width),
              'xmin': int(box[0][1]* im_width),
              'ymax': int(box[1][0]* im_height),
              'ymin': int(box[0][0]* im_height)
          },
          'id': self.category_index[item.label_id+1]['name'],
          'id_index': item.label_id+1,
          'score': float(item.score)
        }
        bboxes.append(bbox)
    return bboxes
  
  def Run(self, image, im_width, im_height, image_id = None):
    # default
    # out = self.engine.DetectWithImage(image,\
    #                                   threshold=0.2,\
    #                                   keep_aspect_ratio=True,\
    #                                   relative_coord=False,\
    #                                   top_k=self.max_num)

    
    

    
    

    if(image_id == None):
      # for pytorch
      out = self.engine.DetectWithImage(image,\
                                      threshold=0.2,\
                                      keep_aspect_ratio=False,\
                                      relative_coord=True,\
                                      top_k=self.max_num)
        
      # print('inference time {}'.format(self.engine.get_inference_time()))
      
      bboxes = self.Get_Results(out,im_width,im_height)
        
      return bboxes, len(out)
    else:
      # for pytorch
      out = self.engine.DetectWithImage(image,\
                                      threshold=self.val_threshold,\
                                      keep_aspect_ratio=False,\
                                      relative_coord=True,\
                                      top_k=self.max_num)
    
      bboxes = self.Get_Results(out,im_width,im_height,self.val_threshold)
    
      
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
