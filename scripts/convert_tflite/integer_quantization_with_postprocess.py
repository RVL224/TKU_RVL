import tensorflow as tf
from PIL import Image
import numpy as np
import glob

# integer_quantization_with_postprocess - Input/Output=float32
save_model_dir = "/tf/minda/github/TKU_RVL/save_models/tensorflow/tensorflow_model/ssdlite_mobilenet_fpn6_mixconv_512_bdd/tflite/"
graph_def_file = save_model_dir+"tflite_graph.pb"

dataset_dir = ["/tf/minda/github/detect_ws/dataset/bdd/BDD_train/JPEGImages"]

## Generating a calibration data set
def representative_dataset_gen():
  image_size = 512
  raw_test_data = []
  
  for name in dataset_dir:
    files = glob.glob(name+"/*.jpg")
    i = 0
    for file in files:
      if(i < 100):
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        image = (np.asarray(image)- np.array([123,117,104])).astype(np.float32)
        image = image[np.newaxis,:,:,:]
        raw_test_data.append(image)

        i += 1
      else:
        break
    
  for data in raw_test_data:
    yield [data]
  
tf.compat.v1.enable_eager_execution()

input_arrays=["normalized_input_image_tensor"]
output_arrays=['TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1', 
               'TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3']

input_tensor={"normalized_input_image_tensor":[1,512,512,3]}

# converter
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, 
                                                      output_arrays,input_tensor)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.allow_custom_ops = True
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()

# # output
with open('ssdlite_mobilenet_mixnet_512_integer_quant_with_postprocess.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - ssdlite_mobilenet_mixnet_512_integer_quant_with_postprocess.tflite")