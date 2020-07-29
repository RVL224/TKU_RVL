# TKU edgetpu (object detection api custom)

## Docker 架設

1. Docker 環境架設 [參考](tutorial/docker_setup.md)
2. Dockerfile 架設與執行 [參考](Docker/README.md)

## 環境  

1. 環境  
    
| tool                | version                   |
|---------------------|---------------------------|
| cuda                | 10.0.130                  |
| cudnn               | 7.6 (1.15)                |
| python              | 3.6.9                     |
| tensorflow-gpu      | 1.15                      |
| protobuf-compiler   | 3.0.0                     |
| openCV              | 3.4.7.28                  |
| openCV-contrib      | 3.4.7.28                  |
| pip                 | 19.3.1 above              |
| python-pil          | 7.1.1                     |
| python-lxml         | 4.5.0                     |
| tqdm                | 4.45.0                    |
| edgetpu_compiler    | 2.1.302470888 above       |
| edgetpu_runtime     | 13                        |
| torch               | 1.2.0                     |
| torchvision         | 0.4.0                     |
| yacs                | 0.1.7                     |    


2. 架設  



## 準備資料集

1. 建立dataset資料夾  

2. 使用框圖程式製作訓練集  

    2.1. labelme
    
3. 轉換成VOC-format Dataset  
    
    3.1. 可使用框圖程式內建轉換或自己寫格式
    3.2. 將生成的資料集放於dataset中
    
4. VOC-format Dataset to csv file  
    * csv 輸出每行格式為 (filename,width,height,class,xmin,ymin,xmax,ymax)  
    
    4.1. 編輯 xml2csv_config.json 
    ```bash
      * label_path: 資料集的標註檔案位置 : (Annotations)
      * out_path: 輸出csv檔案的位置與檔名 : (.csv)
    ```
    
    4.2. 執行
    ```bash
      $ python xml_to_csv.py --config_path <your_xml2csv_config_json_path>
    ```
    
5. csv to tfrecord  
    
    5.1. 編輯 generate_tfrecord_config.json
    ```bash
      * csv_path: 上個步驟輸出的csv檔案位置 : (.csv)
      * img_path: 資料集的圖片路徑 : (JPEGImages)
      * out_path: 輸出record檔案的位置與檔名 : (.record)
    ```
    
    5.2. 編輯 generate_tfrecord.py 中的 categoryText2Int function
    * 需符合dataset class 格式
    ```py

      if label == "bike":
        return 1
      elif label == "bus":
        return 2
         .
         .
         .
    ```
    
    5.3. 執行
    ```bash
        $ python generate_tfrecord.py --config_path <your_generate_tfrecord_config_json_path>
    ```
    
6. Build your pbtxt file, follow the style as below  
    
    * 需符合dataset class 格式 (參考放於 label 資料夾中)
    ```txt
      item{
        id: 1
        name: 'bike'
      }
      item{
        id: 2
        name: 'bus'
      }
         .
         .
         .
    ```

## 讀取 pytorch 權重  

1. 下載 pytorch model  

```bash
    $ cd save_models/pytorch
    $ sh download.sh
```

2. 編輯 config file  
    * 參考放於 cfg/train/
```config

  model {
    ssd {
      # 類別數量 (請根據pytorch模型調整 可以從 yaml 找到調整數值)
      # 計算不包含 background (故參考yaml時，記得 n-1 )
      num_classes: 7
      
      # 模型輸入大小 (請根據pytorch模型調整 可以從 yaml 找到調整數值)
      image_resizer {
        fixed_shape_resizer {
          height: 512
          width: 512
        }
      }
      
      # 選取 backbone 
      feature_extractor {
          # ssd mobilenet v2 fpn mixnet (ssd_mobilenet_v2_custom_v10)
          type: "ssd_mobilenet_v2_custom_v10"
      }
      
      # anchor 參數調整 (請根據pytorch模型調整 可以從 yaml 找到調整數值)
      anchor_generator {
        ssd_anchor_generator {
          # 最後輸出的 feature map layer 數量 
          num_layers: 6
          
          # 使用公式計算
          #min_scale: 0.20000000298
          #max_scale: 0.949999988079
          
          # 自定義
          scales: [0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9]
          aspect_ratios: 1.0
          aspect_ratios: 2.0
          aspect_ratios: 0.5
          aspect_ratios: 3.0
          aspect_ratios: 0.333299994469
        
          height_stride: [8, 16, 32, 64, 128, 256]
          width_stride: [8, 16, 32, 64, 128, 256]
          
          # 將第一個feature map 也使用 aspect_ratios 全部種類計算
          # 如果沒使用 目前程式內 只使用 3 種 (詳細：multiple_grid_anchor_generator.py)
          #reduce_boxes_in_lowest_layer: false
        }
      } 
      post_processing {
        batch_non_max_suppression {
          # 後處理閥值
          score_threshold: 0.01
          iou_threshold: 0.45
          
          # 限制最大偵測數量
          max_detections_per_class: 100
          max_total_detections: 100
        }
        
        # 請參考 pytorch model 最後輸出
        score_converter: SOFTMAX
      }
    }
  }
  
  train_config {
    # 訓練步數 (讀取權重時, 只要調 1 就行了)
    num_steps: 1

    # 批次大小(依照 GPU 記憶體大小調整)
    batch_size: 32 

    # 每步之批次大小 分批計算 loss 後整合更新權重
    sync_replicas: true
    startup_delay_steps: 0.0
    replicas_to_aggregate: 8

    # load cpkt weight (讀取 pytorch model 權重也需要使用)
    fine_tune_checkpoint: "MODEL_CKPT_PATH"
    load_all_detection_checkpoint_vars: true
    fine_tune_checkpoint_type: "detection"
    from_detection_checkpoint: true
  } 
  
  # 訓練和讀權重都會用到, 製作方法可參考準備資料集
  train_input_reader {
    # dataset class label
    label_map_path: "label.pbtxt"

    # load dataset to train
    tf_record_input_reader {
      input_path: "train.record"

      # load multiple record file
      #input_path: ["train_a.record","train_b.record"]
    }
  }

  # 量化訓練 (如果要量化可打開使用，不用的話用註解方式關閉)
  graph_rewriter {
    quantization {
      # 量化統計 根據需求調整 通常等 float模型 穩定再執行
      delay: 1000

      weight_bits: 8
      activation_bits: 8
    }
  }
```   

3. 生成 tensorflow 模型
    * 強調 pytorch model 必須與 tensorflow model "一模一樣" 才能讀取 weight
    * pytorch model 請參考 lufficc pytorch ssd
    * 讀取權重是利用 "pickle 檔" 讀取
    * 通常檔案會放於 save_models/tensorflow/tensorflow_model 中，方便之後不用從讀權重，即可訓練　　
    
    3.1. 編輯 train.bash  
        * 參考放於 src/
    ```bash
      CUDA_VISIBLE_DEVICES=0 python <path_of_train.py>
      
      # 選取顯卡 0, 1, ...
      CUDA_VISIBLE_DEVICES=0

      # 輸出模型位置
      --train_dir=<output_path>

      # 模型參數檔位置
      --pipeline_config_path=<config_path>

      # pytorch 權重位置
      # 通常都會取 weight_tf.pickle
      --pytorch_weight_path=<pickle_path>
        
      # pytorch all layer names (按照 tensorflow model 順序)
      # 通常都會取 layer_name_custom.txt
      --pytorch_layers_path=<path_of_all_layer_name.txt>

      # 讀取 pytorch 權重 (fine_tune_checkpoint 需開啟)
      --load_pytorch=True
    ```  
    
    3.2. 執行
    ```bash
        $ ./train.bash
    ```

4. 將pytorch權重 NCHW 轉換成 NHWC 並生成 pickle file  
    * 詳情請看 tutorial

## Demo tensorflow model  

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

2. 轉換tflite模型  

    2.1. 編輯 transfer.bash 
        * 參考放於　src/
    ```bash
      
      # ===============================
      # 輸出定義圖
      
      CUDA_VISIBLE_DEVICES=0 python <path_of_export_tflite_ssd_graph.py>
        
      # 選取執行顯卡 0, 1, ...
      CUDA_VISIBLE_DEVICES=0
      
      # 訓練模型的參數檔位置
      --pipeline_config_path=<config_path>
        
      # 訓練模型的權重位置
      --trained_checkpoint_prefix=<ckpt_path>
        
      # tflite模型輸出位置
      --output_directory=<output_dir>
      
      # 是否添加後處理
      --add_postprocessing_op＝true
      
      ＃ 最大偵測數量
      --max_detections=30
      
      # ===============================
      # 官方內建 輸出 tflite 模型
      
      tflite_convert 
      
      # tflite模型輸出位置 : (.tflite)
      --output_file
      
      # 輸入定義圖 : (.pb)
      --graph_def_file
      
      # 模型精度 : (FLOAT, QUANTIZED_UINT8)
      --inference_type
      
      # 輸入tensor名稱 (通常: normalized_input_image_tensor)
      --input_arrays
      
      # 輸出tensor名稱 (如果有加入後處理: TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3)
      -output_arrays
      
      # 量化處理需要(對參數影響很大)
      --mean_values
      --std_dev_values
      
      # 輸入tensor的形狀
      --input_shapes
      
      # 允許後處理
      --allow_custom_ops
      
      # 量化不完全可使用 (同常使用map 會大幅下降)
      --default_ranges_min
      --default_ranges_max
      
      # default
      --allow_nudging_weights_to_use_fast_gemm_kernel=true
      --change_concat_input_ranges=false
      
    ```
    
    2.2 執行　　
    ```bash
        $ ./transfer.bash
    ```
    
    2.3 備註
        * 利用post training 進行量化, 請參考 [Tensorflow Lite post-training](https://qiita.com/PINTO/items/008c54536fca690e0572)
    
3. 測試模型  

    3.1. 編輯 detect_process.json  
        * 參考放於　cfg/demo
    ```json
      * PATH_FROZEN_GRAPH : 固化模型位置 (.pb)
      * PATH_TFLITE : tflite 模型位置 (.tflite)
      * PATH_TO_LABELS : dataset class label 位置
      * DATASET_NAME: our, voc case
      * NUM_CLASSES: class num
      * THRESHOLD_BBOX: bounding box 閥值

      * VIDEO_FILE : 測試影片位置
      * IMAGE_DATASET : 測試多張圖片位置
      * SINGE_IMAGE : 測試單張圖片位置

      * RESULT_OUT : 結果儲存位置(資料夾)

      # test mAP
      * USE_07_METRIC : 是否使用 voc 2007 evaluation
      * FP_THRESHOLD : 誤報率的門檻
      * VAL_THRESHOLD : 驗證 bbox 的門檻
      * VAL_MAP :  用於驗證精準度之測試集位置
      * VAL_MAP_OUT : 輸出驗證結果位置(資料夾)

      # 尚未驗證
      * PATH_TPU : edgetpu 模型位置 (.tflite)
    ```
    
    3.2. 執行 demo  
   ```bash
     $ python detect_process.py
        
     # args
       # 參數檔位置
       --config_path = <path_of_detect_process.json>

       # 選擇模型 <graph tflite tpu>
       --engine=graph

       # 測試模式 <video, image, images, map>
       --mode=video

       # 儲存圖片
       --save=false

       # 顯示圖片
       --show=false
    ```
    
    3.3 備註  
    ```txt
        * graph, tflite, edgetpu model (不管是否有量化)
        # demo程式 單純只針對特定幾個模型進行前處理調整，如果當前處理方式改變或有些許差異，就必須調整前處理方式
    ```
    
## 驗證模型

1. 生成驗證資料  

    1.1. 編輯 parse_voc_xml.json 參考
    ```json
      * PATH_TO_DATASET : 驗證資料集位置 (資料夾) (voc 格式)
      * PATH_TO_LABELS : dataset class label 位置
      * NUM_CLASSES : 類別數量 
      * OUT_PATH : 輸出驗證資料位置
    ```  
    
    1.2. 生成
    ```bash
      $ python parse_voc_xml.py
        
      # args
        # 參數檔位置
        --config_path = <path_of_parse_voc_xml.json>
    ```  
    
2. 執行驗證  
    
    2.1. 編輯 detect_process.json 參考 (3.1. 編輯 detect_process.json)
    
    2.2. 執行  
    ```bash
      $ python detect_process.py
        
      # args
        # 參數檔位置
        --config_path = <path_of_detect_process.json>

        # 選擇模型 <graph tflite tpu>
        --engine=graph

        # 驗證模式 map
        --mode=map
    ```
    

## 參考

1. 使用
    * [object detection api](https://github.com/tensorflow/models/tree/master/research/object_detection)
    * [labelme](https://github.com/wkentaro/labelme)
 
2. 教學  
    2.1. 量化達人
    * [Super PINTO](https://twitter.com/PINTO03091)
    * [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo)
    * [Tensorflow Lite post-training](https://qiita.com/PINTO/items/008c54536fca690e0572)
 