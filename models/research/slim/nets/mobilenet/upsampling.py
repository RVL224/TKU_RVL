import tensorflow as tf
from nets.mobilenet import conv_blocks as ops
from nets.mobilenet import mobilenet as lib
import copy

slim = tf.contrib.slim
op = lib.op

@tf.contrib.framework.add_arg_scope
def _fixed_padding(inputs, kernel_size, *args, mode='CONSTANT', **kwargs):
    """
    Pads the input along the spatial dimensions independently of input size.

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
      data_format: The input format ('NHWC' or 'NCHW').
      mode: The mode for tf.pad.

    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if kwargs['data_format'] == 'NCHW':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end], [pad_beg, pad_end]], mode=mode)
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]], mode=mode)
    return padded_inputs

# def _upsample(inputs, out_shape, data_format='NHWC',method='bilinear'):
#     # we need to pad with one pixel, so we set kernel_size = 3
#     inputs = _fixed_padding(inputs, 3, mode='SYMMETRIC')

#     # tf.image.resize_bilinear accepts input in format NHWC
#     if data_format == 'NCHW':
#         inputs = tf.transpose(inputs, [0, 2, 3, 1])

#     if data_format == 'NCHW':
#         height = out_shape[3]
#         width = out_shape[2]
#     else:
#         height = out_shape[2]
#         width = out_shape[1]

#     # we padded with 1 pixel from each side and upsample by factor of 2, so new dimensions will be
#     # greater by 4 pixels after interpolation
#     new_height = height + 4
#     new_width = width + 4

#     inputs = tf.image.resize_bilinear(inputs, (new_height, new_width))

#     # trim back to desired size
#     inputs = inputs[:, 2:-2, 2:-2, :]

#     # back to NCHW if needed
#     if data_format == 'NCHW':
#         inputs = tf.transpose(inputs, [0, 3, 1, 2])

#     inputs = tf.identity(inputs, name='upsampled')
#     return inputs

def _upsample(inputs,scale,name='upsampled',method='bilinear'):
  """ for NHWC """
  height, width = inputs.get_shape().as_list()[1:3]
  if(method == 'bilinear'):
    net = tf.image.resize_bilinear(inputs, [scale*height, scale*width])
  elif(method == 'nearest_neighbor'):
    net = tf.image.resize_nearest_neighbor(inputs, [scale*height, scale*width])
  net = tf.identity(net,name=name)
  return net

def FE_Layer(feature_maps,feature_map_fe,is_training):
  with tf.variable_scope('feature_map_reduce'):
    feature_maps_enhance = []
    for i in range(4):
      scope_name = 'feature_map_enhance_{}'.format(i)
      feature_maps_enhance.append(slim.conv2d(feature_maps[feature_map_fe[i]],\
                                            stride=1,\
                                            padding='VALID',\
                                            kernel_size=[1, 1],\
                                            num_outputs=256,\
                                            normalizer_fn=None,\
                                            activation_fn=None,\
                                            weights_initializer=tf.truncated_normal_initializer(stddev=0.03, mean=0.0),\
                                            weights_regularizer=slim.l2_regularizer(0.00004),\
                                            biases_initializer=tf.zeros_initializer(),\
                                            biases_regularizer=None,\
                                            scope=scope_name,\
                                            trainable=is_training))

  feature_maps_enhance.append(feature_maps[feature_map_fe[4]])
  feature_maps_enhance.append(feature_maps[feature_map_fe[5]])
  feature_maps_enhance.append(feature_maps[feature_map_fe[6]])

  with tf.variable_scope('bilinear_upsample'):
    for i in range(6,0,-1):
      name = 'upsampled_{}'.format(i)
      feature_maps_upsample = _upsample(feature_maps_enhance[i],scale=2,name=name)
      feature_maps[feature_map_fe[i-1]] = tf.add(feature_maps_enhance[i-1],feature_maps_upsample)
  
  return feature_maps

# def FE_Layer(feature_maps,feature_map_fe,is_training):
#   feature_maps_enhance = []
#   for i in range(4):
#     scope_name = 'feature_map_enhance_{}'.format(i)
#     feature_maps_enhance.append(slim.conv2d(feature_maps[feature_map_fe[i]],\
#                                             stride=1,\
#                                             kernel_size=[1, 1],\
#                                             num_outputs=256,\
#                                             activation_fn=tf.nn.relu6,\
#                                             weights_initializer=tf.truncated_normal_initializer(stddev=0.03, mean=0.0),\
#                                             weights_regularizer=slim.l2_regularizer(0.00004),
#                                             scope=scope_name,
#                                             trainable=is_training))

#   feature_maps_enhance.append(feature_maps[feature_map_fe[4]])
#   feature_maps_enhance.append(feature_maps[feature_map_fe[5]])
#   feature_maps_enhance.append(feature_maps[feature_map_fe[6]])

#   with tf.name_scope('bilinear_upsample'):
#     for i in range(6,0,-1):
#       name = 'upsampled_{}'.format(i)
#       feature_maps_upsample = _upsample(feature_maps_enhance[i],2,name)
#       feature_maps[feature_map_fe[i-1]] = tf.add(feature_maps_enhance[i-1],feature_maps_upsample)
  
#   return feature_maps 

def FE_Layer_V2(feature_maps,feature_map_fe,is_training):
  feature_maps_enhance = []
  for i in range(4):
    scope_name = 'feature_map_enhance_{}'.format(i)
    feature_maps_enhance.append(slim.conv2d(feature_maps[feature_map_fe[i]],\
                                            stride=1,\
                                            kernel_size=[1, 1],\
                                            num_outputs=256,\
                                            normalizer_fn=slim.batch_norm,\
                                            activation_fn=tf.nn.relu6,\
                                            weights_initializer=tf.truncated_normal_initializer(stddev=0.03, mean=0.0),\
                                            weights_regularizer=slim.l2_regularizer(0.00004),
                                            scope=scope_name,
                                            trainable=is_training))

  feature_maps_enhance.append(feature_maps[feature_map_fe[4]])
  feature_maps_enhance.append(feature_maps[feature_map_fe[5]])
  feature_maps_enhance.append(feature_maps[feature_map_fe[6]])

  with tf.name_scope('upsample'):
    for i in range(6,0,-1):
      name = 'upsampled_{}'.format(i)
      feature_maps_upsample = _upsample(feature_maps_enhance[i],scale=2,name=name)
      feature_maps[feature_map_fe[i-1]] = tf.add(feature_maps_enhance[i-1],feature_maps_upsample)

  for i in range(6):
    scope_name = 'feature_map_enhance_conv_{}'.format(i)
    feature_maps[feature_map_fe[i]] = slim.conv2d(feature_maps[feature_map_fe[i]],\
                                                  stride=1,\
                                                  kernel_size=[3, 3],\
                                                  num_outputs=256,\
                                                  normalizer_fn=slim.batch_norm,\
                                                  activation_fn=tf.nn.relu6,\
                                                  weights_initializer=tf.truncated_normal_initializer(stddev=0.03, mean=0.0),\
                                                  weights_regularizer=slim.l2_regularizer(0.00004),
                                                  scope=scope_name,
                                                  trainable=is_training)

  return feature_maps

""" FE layer V3 """
def Feature_Map_Expand(feature_maps,feature_map_fe,is_training):
  feature_maps_expand = []
  with tf.variable_scope('feature_map_expand'):
    for i in range(4):
      scope_name = 'expand_{}'.format(i)
      feature_maps_expand.append(slim.conv2d(feature_maps[feature_map_fe[i]],\
                                            stride=1,\
                                            padding='VALID',\
                                            kernel_size=[1, 1],\
                                            num_outputs=256,\
                                            normalizer_fn=None,\
                                            activation_fn=None,\
                                            weights_initializer=tf.truncated_normal_initializer(stddev=0.03, mean=0.0),\
                                            weights_regularizer=slim.l2_regularizer(0.00004),\
                                            biases_initializer=tf.zeros_initializer(),\
                                            biases_regularizer=None,\
                                            scope=scope_name,\
                                            trainable=is_training))

  feature_maps_expand.append(feature_maps[feature_map_fe[4]])
  feature_maps_expand.append(feature_maps[feature_map_fe[5]])
  return feature_maps_expand

def FE_Layer_V3(feature_maps,is_training):
  feature_maps_enhance = [None]*len(feature_maps)
  with tf.variable_scope('FE_Layer'):
    for i in range(5,0,-1):
      name = 'upsampled_{}'.format(5-i)
      feature_maps_upsample = _upsample(feature_maps[i],scale=2,name=name)
      feature_maps_add = tf.add(feature_maps[i-1],feature_maps_upsample)

      scope_name = 'fe_conv_{}'.format(i-1)
      conv_layer = slim.conv2d(feature_maps_add,\
                              stride=1,\
                              kernel_size=[1, 1],\
                              num_outputs=128,\
                              normalizer_fn=slim.batch_norm,\
                              activation_fn=tf.nn.relu6,\
                              weights_initializer=tf.truncated_normal_initializer(stddev=0.03, mean=0.0),\
                              weights_regularizer=slim.l2_regularizer(0.00004),\
                              biases_initializer=None,\
                              biases_regularizer=None,\
                              scope=scope_name,\
                              trainable=is_training)
      
      scope_name = 'fe_mobile_block_{}'.format(i-1)
      feature_maps_enhance[i-1] = ops.expanded_conv(conv_layer,\
                                                    num_outputs = 256,\
                                                    expansion_size=ops.expand_input_by_factor(1,divisible_by=1),\
                                                    scope = scope_name)
    
    feature_maps_enhance[5] = feature_maps[5]
    
    return feature_maps_enhance


# def FE_Layer_V3(feature_maps,is_training):
#   feature_maps_enhance = [None]*len(feature_maps)
#   with tf.variable_scope('FE_Layer'):
#     for i in range(5,0,-1):
#       name = 'upsampled_{}'.format(5-i)
#       feature_maps_upsample = _upsample(feature_maps[i],scale=2,name=name)
#       feature_maps_add = tf.add(feature_maps[i-1],feature_maps_upsample)

#       # scope_name = 'fe_conv_{}_depthwise'.format(i-1)
#       # net = slim.separable_conv2d(feature_maps_add, num_outputs=None, kernel_size=3,
#       #                               depth_multiplier=1, stride=1,
#       #                               rate=1, normalizer_fn=slim.batch_norm,
#       #                               activation_fn=tf.nn.relu6,
#       #                               padding='SAME', scope=scope_name)

#       # scope_name = 'fe_conv_{}_project'.format(i-1)
#       # net = slim.conv2d(net, 256, 1, normalizer_fn=slim.batch_norm,
#       #                   activation_fn=tf.identity, scope=scope_name)
#       # feature_maps_enhance[i-1] = feature_maps_add + net

#       scope_name = 'fe_mobile_block_{}'.format(i-1)
#       feature_maps_enhance[i-1] = ops.expanded_conv(feature_maps_add,\
#                                                     num_outputs = 256,\
#                                                     expansion_size=ops.expand_input_by_factor(1,divisible_by=1),\
#                                                     scope = scope_name)
      
    
#     feature_maps_enhance[5] = feature_maps[5]
    
#     return feature_maps_enhance