import sys
import tensorflow as tf
import numpy as np

class Engine(object):
  def __init__(self,cfg,category_index):
    self.cfg = cfg
    self.category_index = category_index
    self.Init_Graph()
    
    self.val_threshold = cfg['VAL_THRESHOLD']

    """ gpu process """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    """ init sess """
    self.sess = tf.Session(graph=self.graph, config=config)

    """ init setting """
    self.image_tensor, self.out = self.Init_Setting()

  def Init_Graph(self):
    self.graph = tf.Graph()
    with self.graph.as_default():
      od_graph_def = tf.compat.v1.GraphDef()
      with tf.io.gfile.GFile(self.cfg['PATH_FROZEN_GRAPH'], 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
  
  def Init_Setting(self):
    """ input """
    image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
    """ output """
    boxes = self.graph.get_tensor_by_name('detection_boxes:0')
    classes = self.graph.get_tensor_by_name('detection_classes:0')
    scores = self.graph.get_tensor_by_name('detection_scores:0')
    num_detections = self.graph.get_tensor_by_name('num_detections:0')

    out = [boxes, classes, scores, num_detections]

    return image_tensor, out
  
  def Get_Results(self,boxes, classes, scores, im_width, im_height, min_score_thresh=.01):
    bboxes = list()
    for i, box in enumerate(boxes):
      if scores[i] > min_score_thresh:
        ymin, xmin, ymax, xmax = box
        bbox = {
          'bbox': {
              'xmax': int(xmax * im_width),
              'xmin': int(xmin * im_width),
              'ymax': int(ymax * im_height),
              'ymin': int(ymin * im_height)
          },
          'id': self.category_index[classes[i]]['name'],
          'id_index': classes[i],
          'score': float(scores[i])
        }
        bboxes.append(bbox)
    return bboxes
  
  def Run(self, image, im_width, im_height, image_id = None):
    
    boxes, classes, scores, num_detections =  self.sess.run(self.out,\
                    feed_dict={self.image_tensor: image})

    # print(len(boxes)) 
    # print(boxes)
    # print(classes)
    # print(scores)
    # print(num_detections)
      
    if(image_id == None):
      bboxes = self.Get_Results(
               np.squeeze(boxes),\
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
