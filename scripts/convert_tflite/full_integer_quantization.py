import tensorflow as tf
from PIL import Image
import numpy as np
import glob

# full_integer_quantization_with_postprocess - Input/Output=uint8
save_model_dir = "/tf/minda/github/TKU_RVL/out/ssdlite_mobilenet_fpn6_mixconv_512_bdd/tflite/"
graph_def_file = save_model_dir+"tflite_graph.pb"

tf.compat.v1.enable_eager_execution()

input_arrays=["normalized_input_image_tensor"]
output_arrays=['TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1', 
               'TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3']

input_tensor={"normalized_input_image_tensor":[1,512,512,3]}

# converter
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, 
                                                      output_arrays,input_tensor)
converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
converter.allow_custom_ops = True
input_arrays = converter.get_input_arrays()
converter.quantized_input_stats = {input_arrays[0] : (128., 1.)}
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()

# output
with open('ssdlite_mobilenet_mixnet_512_full_integer_quant_with_postprocess.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - ssdlite_mobilenet_mixnet_512_full_integer_quant_with_postprocess.tflite")