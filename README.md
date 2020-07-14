# TKU edgetpu (object detection api custom)

## Docker 架設

## 環境設定

## 準備資料集

1. 使用框圖程式製作訓練集  

    1.1. labelme
    
2. 轉換成VOC-format Dataset  
    
    2.1. 可使用框圖程式內建轉換
    
3. VOC-format Dataset to csv file  
    
    3.1. 編輯 xml2csv_config.json 
    ```bash
      * label_path: 資料集的標註檔案位置 : (Annotations)
      * out_path: 輸出csv檔案的位置與檔名(.csv)
    ```
    
    3.2. 執行
    ```bash
      $ python xml_to_csv.py --config_path <your_xml2csv_config_json_path>
    ```
    
4. csv to tfrecord  
    
    4.1. 編輯 generate_tfrecord_config.json
    ```bash
      * csv_path: 上個步驟輸出的csv檔案位置 (.csv)
      * img_path: 資料集的圖片路徑 : (JPEGImages)
      * out_path: 輸出record檔案的位置與檔名(.record)
    ```
    
    4.2. 編輯 generate_tfrecord.py 中的 categoryText2Int function
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
    
    4.3. 執行
    ```bash
        $ python generate_tfrecord.py --config_path <your_generate_tfrecord_config_json_path>
    ```
    
5. Build your pbtxt file, follow the style as below  
    
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

## 參考

1. 使用
    * [object detection api](https://github.com/tensorflow/models/tree/master/research/object_detection)
    * [labelme](https://github.com/wkentaro/labelme)
 
2. 教學  
    2.1. 量化達人
    * [Super PINTO](https://twitter.com/PINTO03091)
    * [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo)
 