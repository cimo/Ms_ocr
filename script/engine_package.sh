#!/bin/bash

set -euo pipefail

if command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null
then
    isGpu=true
else
    isGpu=false
fi

if [[ "${MS_O_RUNTIME}" == "engine_tesseract" || "${MS_O_RUNTIME}" == "engine_realtime" ]]
then
    python3 -m pip uninstall -y paddlepaddle-gpu paddlepaddle >/dev/null 2>&1 || true

    if [ ${isGpu} = true ]
    then
        python3 -m pip install --break-system-packages --ignore-installed torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
    else
        python3 -m pip install --break-system-packages --ignore-installed torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
else
    python3 -m pip uninstall -y torch torchvision >/dev/null 2>&1 || true

    if [ ${isGpu} = true ]
    then
        python3 -m pip install --break-system-packages --ignore-installed paddlepaddle-gpu==3.3.0 --index-url https://www.paddlepaddle.org.cn/packages/stable/cu129
    else
        python3 -m pip install --break-system-packages --ignore-installed paddlepaddle==3.3.0 --index-url https://www.paddlepaddle.org.cn/packages/stable/cpu
    fi
fi
