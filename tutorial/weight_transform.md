#  權重轉換 (NCHW to NHWC)

## Environment

| tool                | version                   |
|---------------------|---------------------------|
| torch               | 1.2.0                     |
| torchvision         | 0.4.0                     |
| nvidia driver       | >= 430.64                 |
| CUDA                | 10.0.130                  |
| cudnn               | 7.6                       |


## 步驟  

1. 得到 pytorch 權重與參數檔  
2. 執行 weight_transform.py (check_torch 模式) 生成 layer_name.txt  
3. 編輯 Tensorflow model 參數檔 (config file)  
4. 執行 train.bash (check_model 模式) 生成 layer_name_tf.txt  
5. 比對模型 使模型運算(layer_name.txt)順序達到跟layer_name_tf.txt 一樣 (可生成另一個存放比對後的結果 (layer_name_custom.txt))  
6. 執行 weight_transform.py (save pickle 模型) 將 pytorch 權重資料格式 NCHW 轉換成 NHWC 並生成 pickle file  
7. 執行 train.bash (load_pytorch 模型) 生成 tensorflow model (.pb)  


## 查詢 pytorch ops name 和 轉換 pytorch 權重資料格式 之程式
  
  * 尚未優化
  * weight_transform.py
  * conv format : NCHW to NHWC

```bash
  # check_model function
    * usage : 用來跟 tensorflow model 對照 (查看並儲存 pytorch model all layer names)
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
    * check_torch : [bool] 查看並儲存 pytorch model all layer names 用來跟 tensorflow model 比較 (<weight_pth_path>/layer_name.txt)
    * layer_name_torch : 排序過(pytorch model 和 tensorflow model 每層都需對應無誤)的所有層名稱之 layer_name_custom.txt 
    * layer_name_tf : tensorflow model all layer names
    * save_path : pickle path
```

## Reference
  * [lufficc pytorch ssd](https://github.com/lufficc/SSD?fbclid=IwAR2WFi1g6gbpH8GzSBBO-ERHTUIX7VXbPbTtK5Z-kIT1h-dSWlx3GEHkkqc)
