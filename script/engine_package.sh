#!/bin/bash

if command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null
then
    isGpu=true
else
    isGpu=false
fi

if [ ${MS_O_ENGINE} = "tesseract" ]
then
    python3 -m pip uninstall -y paddlepaddle-gpu paddlepaddle >/dev/null 2>&1

    if [ ${isGpu} = true ]
    then
        python3 -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu126
    else
        python3 -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cpu
    fi
else
    python3 -m pip uninstall -y torch torchvision >/dev/null 2>&1

    if [ ${isGpu} = true ]
    then
        python3 -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu129
    else
        python3 -m pip install paddlepaddle==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu
    fi
fi
