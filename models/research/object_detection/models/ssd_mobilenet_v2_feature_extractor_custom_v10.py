# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""SSDFeatureExtractor for MobilenetV2 features."""

# pytorch ssdlite mobilenet v2 mixnet anchor 3 
# (featuremap: backbone 7 to output 6)
import sys
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim

from object_detection.meta_architectures import ssd_meta_arch
# from object_detection.models import feature_map_generators
# from object_detection.models import feature_map_generators_custom_for_v5 as feature_map_generators
from object_detection.models import feature_map_generators_custom_for_v10 as feature_map_generators

from object_detection.utils import context_manager
from object_detection.utils import ops
from object_detection.utils import shape_utils

from nets.mobilenet import mobilenet
from nets.mobilenet import mobilenet_v2
from object_detection.core import preprocessor

slim = contrib_slim

global custom_modify_list, feed_dict_list

class SSDMobileNetV2CustomV10FeatureExtractor(ssd_meta_arch.SSDFeatureExtractor):
  """SSD Feature Extractor using MobilenetV2 features."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams_fn,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False,
               num_layers=7,
               override_base_feature_extractor_hyperparams=False):
    """MobileNetV2 Feature Extractor for SSD Models.

    Mobilenet v2 (experimental), designed by sandler@. More details can be found
    in //knowledge/cerebra/brain/compression/mobilenet/mobilenet_experimental.py

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams_fn: A function to construct tf slim arg_scope for conv2d
        and separable_conv2d ops in the layers that are added on top of the
        base feature extractor.
      reuse_weights: Whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False.
      use_depthwise: Whether to use depthwise convolutions. Default is False.
      num_layers: Number of SSD layers.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.
    """
    super(SSDMobileNetV2CustomV10FeatureExtractor, self).__init__(
        is_training=is_training,
        depth_multiplier=depth_multiplier,
        min_depth=min_depth,
        pad_to_multiple=pad_to_multiple,
        conv_hyperparams_fn=conv_hyperparams_fn,
        reuse_weights=reuse_weights,
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        num_layers=num_layers,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams)

  def preprocess(self, resized_inputs):
    """SSD preprocessing.

    Maps pixel values to the range [-1, 1].

    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    # return (2.0 / 255.0) * resized_inputs - 1.0
    # return resized_inputs
    return  preprocessor.subtract_channel_mean_custom(resized_inputs,[123, 117, 104])

    # resized_inputs = preprocessor.subtract_channel_mean_custom(resized_inputs,[123, 117, 104])
    # return  (2.0 / 255.0) * resized_inputs - 1.0


  def extract_features(self, preprocessed_inputs):
    """Extract features from preprocessed inputs.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      feature_maps: a list of tensors where the ith tensor has shape
        [batch, height_i, width_i, depth_i]
    """
    preprocessed_inputs = shape_utils.check_min_image_dim(
        33, preprocessed_inputs)

    # batchnorm param
    
    # normalizer_params = {
    #   'is_training': False,
    #   'decay': 0.9, 
    #   'center': True,
    #   'scale': True,
    #   "epsilon": 0.00001
    # }

    normalizer_params = {
      'is_training': self._is_training,
      'decay': 0.9, 
      'center': True,
      'scale': True,
      "epsilon": 0.00001
    }

    # feature map
    feature_map_layout = {
        # 'from_layer': ['layer_14', 'layer_19', '', '', '',''][:6],
        # 'layer_depth':[-1, -1, 512, 256, 256, 256][:6],   
        'from_layer': ['layer_7','layer_14', 'layer_19', '', '', '',''][:self._num_layers],
        'layer_depth':[-1, -1, -1, 512, 256, 256, 256][:self._num_layers],
        'use_depthwise': self._use_depthwise,
        'use_explicit_padding': self._use_explicit_padding,
    }

    #  feature map fpn
    feature_fpn_blocks = [
      'layer_7',
      'layer_14',
      'layer_19',
      'layer_19_2_Conv2d_3_s2_512',
      'layer_19_2_Conv2d_4_s2_256',
      'layer_19_2_Conv2d_5_s2_256',
      'layer_19_2_Conv2d_6_s2_256'
    ]

    # fpn param
    additional_layer_depth = 256
    use_native_resize_op = True

    with tf.variable_scope('MobilenetV2', reuse=self._reuse_weights) as scope:
      with slim.arg_scope(
          mobilenet_v2.training_scope(is_training=None, bn_decay=0.9997)), \
          slim.arg_scope(
              [mobilenet.depth_multiplier], min_depth=self._min_depth):
        with (slim.arg_scope(self._conv_hyperparams_fn())
              if self._override_base_feature_extractor_hyperparams else
              context_manager.IdentityContextManager()):
          with (slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                              normalizer_params=normalizer_params)):
            _, image_features = mobilenet_v2.mobilenet_base(
                ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple),
                final_endpoint='layer_19',
                depth_multiplier=self._depth_multiplier,
                use_explicit_padding=self._use_explicit_padding,
                scope=scope)
        with slim.arg_scope(self._conv_hyperparams_fn()):
          # with (slim.arg_scope([slim.conv2d, slim.separable_conv2d],
          #                     normalizer_params=normalizer_params)):
          feature_maps = feature_map_generators.multi_resolution_mixnet_feature_maps(
              feature_map_layout=feature_map_layout,
              depth_multiplier=self._depth_multiplier,
              min_depth=self._min_depth,
              insert_1x1_conv=True,
              # insert_1x1_conv=False,
              image_features=image_features,
              mix_num=4)

        # fpn
        depth_fn = lambda d: max(int(d * self._depth_multiplier), self._min_depth)
        with slim.arg_scope(self._conv_hyperparams_fn()):
          with (slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                                normalizer_fn=slim.batch_norm,
                                 normalizer_params=normalizer_params)):
            with tf.variable_scope('fpn', reuse=self._reuse_weights):
              fpn_features = feature_map_generators.fpn_top_down_feature_maps(
                [(key, feature_maps[key]) for key in feature_fpn_blocks],
                depth=depth_fn(additional_layer_depth),
                use_depthwise=self._use_depthwise,
                use_explicit_padding=self._use_explicit_padding,
                use_native_resize_op=use_native_resize_op,
                last_feature_map_use=False)
        
        # for item in fpn_features.items():
        #   print(item)
        # sys.exit()

    return fpn_features.values()
    # return feature_maps.values()
