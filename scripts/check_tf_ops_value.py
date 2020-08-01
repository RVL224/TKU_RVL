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
  
      # output tensor
      # output_tensor = detection_graph.get_tensor_by_name('FeatureExtractor/MobilenetV2/MobilenetV2/input:0')
      output_tensor = detection_graph.get_tensor_by_name('FeatureExtractor/MobilenetV2/expanded_conv_5/output:0')
    
      # image preprocess
      frame = cv2.imread(image_file,cv2.IMREAD_COLOR)
      frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
      im_height, im_width, _ = frame.shape
      frame_expanded = np.expand_dims(frame_rgb, axis=0)
    
      # run
      out = sess.run(output_tensor, feed_dict={input_tensor: frame_expanded})
    
      # print tensor value
      print("tensor shape {}".format(out.shape))
      print("tensor value")
      print(out)
    

def main():
  args = parse_args()
  print(args)
    
  run(model_path=args.model_path,
      image_file=args.image_file)

if __name__ == '__main__':
  main()