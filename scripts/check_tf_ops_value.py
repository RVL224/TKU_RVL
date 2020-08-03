""" sys """
import sys
import os
import argparse
import numpy as np
import cv2

""" tf """
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf

def parse_args():
  parser = argparse.ArgumentParser(description='TF model graph')
  parser.add_argument("--model_path",\
                      type = str,\
                      default="",\
                      help = "tensorflow frozen model path")
  
  parser.add_argument("--image_file",\
                      type = str,\
                      default="",\
                      help = "image path")
  
  return parser.parse_args()

def Get_Results(boxes, classes, scores, im_width, im_height, min_score_thresh=.2):
  bboxes = list()
  for i, box in enumerate(boxes):
    if scores[i] > min_score_thresh:
      # ymin, xmin, ymax, xmax = box
      xmin, ymin, xmax, ymax = box
      bbox = {
        'bbox': {
          'xmax': int(xmax * im_width),
          'xmin': int(xmin * im_width),
          'ymax': int(ymax * im_height),
          'ymin': int(ymin * im_height)
        },
        'id': str(classes[i]),
        'score': float(scores[i])
      }
      bboxes.append(bbox)
  return bboxes

def draw_BBox(frame,bboxes,min_score_thresh = 0.2):
  for bbox in bboxes:
    if(bbox['score'] >= min_score_thresh):
      cv2.rectangle(frame,\
                    (bbox['bbox']['xmin'], bbox['bbox']['ymax']),\
                    (bbox['bbox']['xmax'], bbox['bbox']['ymin']),\
                    (0, 255, 0), 2)
      
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
                  font, 1, (0, 255, 0), 1, cv2.LINE_AA)

def run(model_path, image_file):
  
  """ load graph """
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(model_path, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
  
  with detection_graph.as_default():
    with tf.Session() as sess:
      # input tensor
      input_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      
      """ image preprocess """
      frame = cv2.imread(image_file,cv2.IMREAD_COLOR)
      frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
      im_height, im_width, _ = frame.shape
      frame_expanded = np.expand_dims(frame_rgb, axis=0)
    
      """ 
        output tensor
          preprocess
          mobilenet_featupremap_0
          ClassPredictor_0
          BoxPredictor_0
          ClassPredictor_concat
          BoxPredictor_concat
      """
      # preprocess
      # output_tensor = detection_graph.get_tensor_by_name('FeatureExtractor/MobilenetV2/MobilenetV2/input:0')
        
      # mobilenet_featupremap_0
      # output_tensor = detection_graph.get_tensor_by_name('FeatureExtractor/MobilenetV2/expanded_conv_5/output:0')
        
      # ClassPredictor_0
      # output_tensor = detection_graph.get_tensor_by_name('BoxPredictor_0/ClassPredictor/BiasAdd:0')
        
      # BoxPredictor_0
      # output_tensor = detection_graph.get_tensor_by_name('BoxPredictor_0/BoxEncodingPredictor/BiasAdd:0')
        
      # ClassPredictor_concat
      # output_tensor = detection_graph.get_tensor_by_name('concat_1:0')
        
      # BoxPredictor_concat
      # output_tensor = detection_graph.get_tensor_by_name('concat:0')
        
      # out = sess.run(output_tensor, feed_dict={input_tensor: frame_expanded})
      # print("tensor shape {}".format(out.shape))
      # print("tensor value")
      # print(out)
      
      """ 
        anchor : 程式執行順序, 將影響 tensor 名稱順序 (可以指定tensor名稱解決)
        reference : models/research/object_detection/core/box_list.py
          function: get_center_coordinates_and_sizes
      """
      # w_tensor = detection_graph.get_tensor_by_name('Postprocessor/Decode/get_center_coordinates_and_sizes/clip_by_value:0')
      # h_tensor = detection_graph.get_tensor_by_name('Postprocessor/Decode/get_center_coordinates_and_sizes/clip_by_value_1:0')
      # cy_tensor = detection_graph.get_tensor_by_name('Postprocessor/Decode/get_center_coordinates_and_sizes/clip_by_value_2:0')
      # cx_tensor = detection_graph.get_tensor_by_name('Postprocessor/Decode/get_center_coordinates_and_sizes/clip_by_value_3:0')
        
      # w = sess.run(w_tensor,feed_dict={input_tensor: frame_expanded})
      # h = sess.run(h_tensor,feed_dict={input_tensor: frame_expanded})
      # cy = sess.run(cy_tensor,feed_dict={input_tensor: frame_expanded})
      # cx = sess.run(cx_tensor,feed_dict={input_tensor: frame_expanded})
      # anchors = np.stack([cx,cy,w,h],axis=1)
      # print(anchors.shape)
      # print(anchors)
        
      """ 
        bounding box 
          x, y, w, h
          xmin, ymin, xmax, ymax
          
        # reference : models/research/object_detection/box_coders/faster_rcnn_box_coder.py
        # fucntion : _decode
        # 程式執行順序, 將影響 tensor 名稱順序 (可以指定tensor名稱解決)
        # 最需注意: 執行結果順序 bbox(x, y, w, h) 跟轉移權重的模型有關 (目前使用肉眼比對解決)
      """
      # x, y, w, h
      # bw_tensor = detection_graph.get_tensor_by_name('Postprocessor/Decode/mul:0')
      # bh_tensor = detection_graph.get_tensor_by_name('Postprocessor/Decode/mul_1:0')
      # bcy_tensor = detection_graph.get_tensor_by_name('Postprocessor/Decode/add:0')
      # bcx_tensor = detection_graph.get_tensor_by_name('Postprocessor/Decode/add_1:0')
   
      # w = sess.run(bw_tensor,feed_dict={input_tensor: frame_expanded})
      # h = sess.run(bh_tensor,feed_dict={input_tensor: frame_expanded})
      # cy = sess.run(bcy_tensor,feed_dict={input_tensor: frame_expanded})
      # cx = sess.run(bcx_tensor,feed_dict={input_tensor: frame_expanded})
        
      # bbox_location = np.stack([cx,cy,w,h],axis=1)
      # print(bbox_location.shape)
      # print(bbox_location)
        
      # xmin, ymin, xmax, ymax  
      # bymin_tensor = detection_graph.get_tensor_by_name('Postprocessor/Decode/sub:0')
      # bxmin_tensor = detection_graph.get_tensor_by_name('Postprocessor/Decode/sub_1:0')
      # bymax_tensor = detection_graph.get_tensor_by_name('Postprocessor/Decode/add_2:0')
      # bxmax_tensor = detection_graph.get_tensor_by_name('Postprocessor/Decode/add_3:0')
   
      # ymin = sess.run(bymin_tensor,feed_dict={input_tensor: frame_expanded})
      # xmin = sess.run(bxmin_tensor,feed_dict={input_tensor: frame_expanded})
      # ymax = sess.run(bymax_tensor,feed_dict={input_tensor: frame_expanded})
      # xmax = sess.run(bxmax_tensor,feed_dict={input_tensor: frame_expanded})
      # bbox_location = np.stack([xmin,ymin,xmax,ymax],axis=1)
    
      # print(bbox_location.shape)
      # print(bbox_location)
      
      """ 
        detection 
          bbox
          score
          class
      """
      boxes_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
      classes_tensor = detection_graph.get_tensor_by_name('detection_classes:0')
      scores_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
        
      boxes, classes, scores = sess.run([boxes_tensor, classes_tensor,scores_tensor],
                                        feed_dict={input_tensor: frame_expanded})
      
      bboxes = Get_Results(np.squeeze(boxes),\
                           np.squeeze(classes).astype(np.int32),\
                           np.squeeze(scores),\
                           im_width,
                           im_height,
                           min_score_thresh=.5)
     
      
def main():
  args = parse_args()
  print(args)
    
  run(model_path=args.model_path,
      image_file=args.image_file)

if __name__ == '__main__':
  main()