#  權重轉換 (NCHW to NHWC)

## Environment

| tool                | version                   |
|---------------------|---------------------------|
| torch               | 1.2.0                     |
| torchvision         | 0.4.0                     |
| nvidia driver       | >= 430.64                 |
| CUDA                | 10.0                      |
| cudnn               | >= 10.0.130               |


## code 解釋
  
  * 尚未優化
  * weight_transform.py
  * conv format : NCHW to NHWC

```bash
  # check_model function
    * usage : 用來跟 tensorflow model 對照 (可以先複製到 layer_name.txt)
    * input 
      * cfg : param.yaml
      * ckpt : pytorch_model_weight.pth
      
    * output
      * print model all layers
  
  # read_layer_name_from_file function
    * usage : 讀取以排序過(pytorch model 和 tensorflow model 每層都需對應無誤)的 layer_name_custom.txt 的所有層名稱
    * input 
      * layer_name_custom.txt
    * return 
      * all layer names list
    
  # get_depth_wise function
    * usage : 讀取 tensorflow model depth_wise conv layer order (depth_wise_conv 轉換方式與 conv 不同) 
    * input
      * layer_name_tf.txt
    * return 
      * tensorflow model's depth_wise conv order list
  
  # save_layer_param function
    * usage : 將 weight (NCHW) to weight (HHWC) and save pickle 
    * input 
      * cfg : param.yaml
      * ckpt : pytorch_model_weight.pth
      * save_path : weight_tf.pickle
      * depth_wise : tensorflow model's depth_wise conv order list
      * layer_names : pytorch model's all layer names list
    * output
      * save param pickle
```

## Run

```bash
  $ cd <worksace>/src
  $ ./weight_transform.bash
  
  * weight_transform.py args
    * config_file : pytorch model param yaml
    * ckpt : pytorch model weight pth
    * check_torch : [bool] 查看並儲存 pytorch model all layer names 用來跟 tensorflow model 比較 (layer_name.txt)
    * layer_name_torch : 排序過(pytorch model 和 tensorflow model 每層都需對應無誤)的所有層名稱之 layer_name_custom.txt 
    * layer_name_tf : tensorflow model all layer names
    * save_path : pickle path
```

## Reference
  * [lufficc pytorch ssd](https://github.com/lufficc/SSD?fbclid=IwAR2WFi1g6gbpH8GzSBBO-ERHTUIX7VXbPbTtK5Z-kIT1h-dSWlx3GEHkkqc)
