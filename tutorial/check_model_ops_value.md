# 檢查模型運算輸出

* 檢查權重轉移時，不要使用 Quantization aware training (add fake quant node)

## Pytorch  

1. 檢查前處理與最後模型輸出程式  

    1.1 程式解釋   
    ```python
      # run_demo_image function
        * usage : 用於模型運算輸出檢查
        * input :
          * cfg : pytorch model 參數
          * ckpt : pytorch model 權重
          * score_threshold : Bounding Box 門檻值
          * image_name : 圖片檔案位置
        * ouput : 根據 check_tensor_value function 使用而有所不同
      
      # check_tensor_value function
        * usage : 用於檢查運算輸出
        * input : 
          * tensor : 欲顯示的張量
          * mode : 
            * tf : 將輸出格式轉換跟 tensorflow model 一樣
        * output:
          * print shape and value 
    ```  
    
    1.2 執行  
    ```bash
      $ cd <worksace>/src
      $ ./pytorch_demo.bash
      
      # pytorch_demo.py args
        * cfg : pytorch model 參數
        * ckpt : pytorch model 權重
        * score_threshold : Bounding Box 門檻值
        * image_name : 圖片檔案位置
    ```
    
## TensorFlow  

1. 固化模型  

    1.1. 編輯 frozen_graph.bash  
        * 參考放於　src/
    ```bash
      CUDA_VISIBLE_DEVICES=0 python <path_of_export_inference_graph.py>
        
      # 選取顯卡 0, 1, ...
      CUDA_VISIBLE_DEVICES=0
      
      # 訓練模型的參數檔位置
      --pipeline_config_path=<config_path>
        
      # 訓練模型的權重位置
      --trained_checkpoint_prefix=<ckpt_path>
        
      # 固化模型輸出位置
      --output_directory=<output_dir>
    ```
    
    1.2 執行　　
    ```bash
        $ ./frozen_graph.bash
    ```   
    
2. 輸出運算名稱和可視化圖  

    2.1. 編輯 produce_tf_model_graph.bash  
        * 參考放於 src/
    ```bash
      CUDA_VISIBLE_DEVICES=0 python3 <path_of_produce_tf_model_graph.py>
      
      # 選取顯卡 0, 1, ...
      CUDA_VISIBLE_DEVICES=0
      
      # 模型位置
      --model_path=<.pb_path>
      
      # 是否輸出 TensorBoard
      --save=<bool>
      
      # Tensorboard 輸出位置
      --output_dir=<output_dir>
    ```  
    
    2.2 執行　　
    ```bash
        $ ./produce_tf_model_graph.bash
    ```   
    
    2.3 利用 Tensorboard 查看模型建構和運算
    
    ```bash
      $ cd <path_of_Tensorboard_output_dir>
      
      $ tensorboard --logdir .
      
      # 點擊終端機上的網址
      # 注意 Docker 沒開指定的port無法執行，所以可以載下來用本機端的 tensorboard 觀看
    ```
    
3. 檢查整體模型運算輸出之程式  

    3.1 編輯 check_tf_ops_value.bash  
        * 參考放於 src/
    ```bash
      CUDA_VISIBLE_DEVICES=0 python3 <path_of_check_tf_ops_value.py>
      
      # 模型位置
      --model_path=<path_of_pb_file>
      
      # 圖片位置
      --image_file=<path_of_image>
    
    ```  
    
    3.2 程式解釋 check_tf_ops_value.py  
    
    ```python
      # run function
        * usage : 用於模型運算輸出檢查
        * input :
          * model_path : 模型位置
          * image_file : 圖片檔案位置
        * ouput : 根據 output_tensor 使用而有所不同
      
      # 修改 output_tensor 中的運算名稱 (可利用tensorboard)
      output_tensor = detection_graph.get_tensor_by_name('<ops_name>:0')
    ```  
    
    3.3 執行　　
    ```bash
        $ ./check_tf_ops_value.bash
    ``` 


