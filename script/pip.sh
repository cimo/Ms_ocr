#!/bin/bash

pathEnv="${PATH_ROOT}.venv/"

if [ ! -d "${pathEnv}" ]
then
    python3 -m venv "${pathEnv}"
fi

. "${pathEnv}bin/activate"

python3 -m pip install -r "${PATH_ROOT}requirement.txt"

# Onnx
cpuVendor=$(awk -F: '/vendor_id/{gsub(/^[ \t]+/,"",$2); print $2; exit}' /proc/cpuinfo)

if [ "${cpuVendor}" = "AuthenticAMD" ]
then
    python3 -m pip install onnxruntime==1.24.4
elif [ "$cpuVendor" = "GenuineIntel" ]
then
    python3 -m pip install onnxruntime-openvino==1.24.1
fi

# Onnx - document_scanner
pathModel="/home/app/onnx/document_scanner/model/"
urlModel="https://huggingface.co/cimo001/paddle/resolve/main/"

mkdir -p "${pathModel}"

modelList=(
    "PP-DocLayout_plus-L/onnx/pp-docLayout_plus-l.onnx"
    "PP-OCRv6_medium_det/onnx/pp-ocrV6_medium_det.onnx"
    "PP-OCRv6_medium_rec/onnx/pp-ocrV6_medium_rec.onnx"
    "PP-OCRv6_medium_rec/onnx/dictionary.txt"
)

for model in "${modelList[@]}"
do
    fileName=$(basename "${model}")

    if [ ! -f "${pathModel}${fileName}" ]
    then
        echo "Download document_scanner: ${fileName}"

        if ! curl -fsSL "${urlModel}${model}" -o "${pathModel}${fileName}"
        then
            echo "Skip document_scanner - ${fileName}: download failed."

            rm -f "${pathModel}${fileName}"
        fi
    fi
done

python3 "${PATH_ROOT}onnx/document_scanner/server.py" >> "${PATH_ROOT}${MS_O_PATH_LOG}document_scanner.log" 2>&1 &
