#!/bin/sh

echo "=== Choose pytorch model ==="
echo "(1) ssdlite_mobilenet_fpn6_mixconv_512"
echo "============================\n"

read -p "Input number for choose model? " choose_model

case $choose_model in
    "1") 
        echo "choose (${choose_model}) ssdlite_mobilenet_fpn6_mixconv_512 \n"
        model_folder="ssdlite_mobilenet_fpn6_mixconv_512"
        file_id="10PapR0BQGE6GGKRXH3xoovBzHWz_SRB0"
        ;;
    *) 
        echo "not found"
        exit 0
        ;;
esac

echo "=== check pytorch_model folder ==="
if [ ! -d "pytorch_model" ]; then
    echo "no exit & mkdir pytorch_model \n"
    mkdir pytorch_model
else
    echo "exist pytorch_model \n"
fi

echo "=== check ${model_folder} ==="
if [ ! -d "pytorch_model/${model_folder}" ]; then

    echo "=== download pytorch model weight file ==="
    
    # download model from google drvie
    curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${file_id}" > /dev/null
    CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
    curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${file_id}" -o ${model_folder}.zip
    
    # unzip & remove
    unzip ${model_folder}.zip
    rm -r ${model_folder}.zip
        
    # move folder
    echo "mv ${model_folder} pytorch_model"
    mv ${model_folder} pytorch_model
    echo "=== success ==="

else
    echo "exist ${model_folder} folder"
fi