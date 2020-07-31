""" sys """
import sys
import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
""" tf """
import tensorflow as tf

def parse_args():
  parser = argparse.ArgumentParser(description='TF model graph')
  parser.add_argument("--model_path",\
                      type = str,\
                      default="",\
                      help = "tensorflow frozen model path")
  
  parser.add_argument("--save",\
                      type = bool,\
                      default=False,\
                      help = "weather to save graph for tensorboard.")
    
  parser.add_argument("--output_dir",\
                      type = str,\
                      default="",\
                      help = "output folder path")

  return parser.parse_args()

def run(model_path, save, output_dir):
  
  """ load graph """
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(model_path, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
  
  with detection_graph.as_default():
    # save graph
    if(save):
      writer = tf.compat.v1.summary.FileWriter(output_dir,graph=detection_graph)
    
    # print tensorflow model ops
    ops = tf.compat.v1.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    for item in all_tensor_names:
      print(item,detection_graph.get_tensor_by_name(item).dtype)

def main():
  args = parse_args()
  print(args)
    
  run(model_path=args.model_path,
      save=args.save,
      output_dir=args.output_dir)

if __name__ == '__main__':
  main()