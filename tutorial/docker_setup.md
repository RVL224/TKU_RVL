# docker 環境建置

## Environment

| tool                | version                   |
|---------------------|---------------------------|
| nvidia-driver       | >= 430.64                 |

## Docker install

```bash

  # uninstall old version
  $ sudo apt-get remove docker docker-engine docker.io containerd runc
  
  # install dependency package
  $ sudo apt-get update
  $ sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
  
  # Add Docker’s official GPG key:
  $ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
  $ sudo apt-key fingerprint 0EBFCD88
  
  # version for x86_64/amd64
  $ sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
  
  # install latest version
  $ sudo apt-get update
  $ sudo apt-get install docker-ce docker-ce-cli containerd.io
  
  # check version
  # sudo docker -v
 
  # to remove the requirement of “sudo” when running docker commands, add your user to the docker group.
  # maybe fail (I do not know why)
  $ sudo usermod -aG docker ${USER}
  
  # Your username is now part of the docker group. To apply changes, either logout and login or type:
  $ su - ${USER}
```

## NVIDIA Dokcer install

```bash
  # add nvidia-docker's official GPG key:
  $ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
  $ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
  $ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
  
  # install latest version
  $ sudo apt-get update && sudo apt-get install -y nvidia-docker2
  $ sudo pkill -SIGHUP dockerd
  
  # check success
  $ docker run --runtime=nvidia --rm nvidia/cuda:9.0-base nvidia-smi
```

## Example start docker container

```
  # download image from docker hub
  $ sudo docker pull tensorflow/tensorflow:1.15.2-gpu-jupyter
  
  # check images
  $ sudo docker images
  
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

## Reference
   
  * [docker offical](https://docs.docker.com/engine/install/ubuntu/)
  * [nvidia-docker install](https://cnvrg.io/how-to-setup-docker-and-nvidia-docker-2-0-on-ubuntu-18-04/)
  * [tensorflow docker run](https://qiita.com/hrappuccino/items/fe76e2ed014c16171e47)
