# Dockerfile 建置

## 建置 Image  

```bash
    $ cd <Dockerfile_path>
    $ sudo docker build --tag=<image_name> .
```

## 執行  

```bash

    # run container
    $ sudo docker run -it --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=0 -p 8888:8888 -name <container_name> -v <host_share_folder_path>:/tf/<share_folder_path> -shm-size='64g' <image_name or image_id>
  
  # args
    * -it => keep running
    * --runtime nvidia => 指定 nvidia dirver
    * -e NVIDIA_VISIBLE_DEVICES=0 => 指定 GPU , 0 為第1顆 GPU
    * -p <open_port>:<host_post> => 指定 (前: 對外可進入的port 後: 本機開放的port)
    * -name <container_name> => 指定 contrainer 名稱
    * -v <host_share_folder_path>:/tf/<share_folder_path> => 指定分享資料夾位置(前: 本機檔案位置 後: contrainer檔案位置)
```