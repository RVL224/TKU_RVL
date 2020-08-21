"""
  Post-training dynamic range quantization
    * dataset : no dataset
    * supported hardware: CPU, GPU(Android)
    * input/output type: float32
    * only weight quantization (8-bits of precision) 
"""
import tensorflow as tf

tf.compat.v1.enable_eager_execution()



# Weight Quantization - Input/Output=float32
save_model_dir = "/tf/minda/github/TKU_RVL/save_models/tensorflow/tensorflow_model/ssdlite_mobilenet_fpn6_mixconv_512_bdd/tflite/"
graph_def_file = save_model_dir+"tflite_graph.pb"

input_arrays=["normalized_input_image_tensor"]
output_arrays=['TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1', 
               'TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3']

input_tensor={"normalized_input_image_tensor":[1,512,512,3]}

# converter
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, 
                                                      output_arrays,input_tensor)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

converter.allow_custom_ops = True
tflite_quant_model = converter.convert()

# output
with open('ssdlite_mobilenet_mixnet_512_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - ssdlite_mobilenet_mixnet_512_weight_quant.tflite")