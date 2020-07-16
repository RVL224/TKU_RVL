# TKU edgetpu (object detection api custom)

## Docker 架設

## 環境設定

## 準備資料集

1. 建立dataset資料夾  

2. 使用框圖程式製作訓練集  

    2.1. labelme
    
3. 轉換成VOC-format Dataset  
    
    3.1. 可使用框圖程式內建轉換  
    3.2. 將生成的資料集放於dataset中
    
4. VOC-format Dataset to csv file  
    
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
  
    3.1. 編輯 train.bash
    ```bash
      python <path_of_train.py>

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

4. 將pytorch權重 NCHW 轉換成 NHWC 並生成 pickle file  
    * 詳情請看 tutorial

## 參考

1. 使用
    * [object detection api](https://github.com/tensorflow/models/tree/master/research/object_detection)
    * [labelme](https://github.com/wkentaro/labelme)
 
2. 教學  
    2.1. 量化達人
    * [Super PINTO](https://twitter.com/PINTO03091)
    * [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo)
 