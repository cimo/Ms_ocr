---
license: apache-2.0
library_name: PaddleOCR
language:
- en
- zh
pipeline_tag: image-to-text
tags:
- OCR
- PaddlePaddle
- PaddleOCR
- layout_detection
---

# PP-DocLayout_plus-L

## Introduction

A higher-precision layout area localization model trained on a self-built dataset containing Chinese and English papers, PPT, multi-layout magazines, contracts, books, exams, ancient books and research reports using RT-DETR-L. The layout detection model includes 20 common categories: document title, paragraph title, text, page number, abstract, table, references, footnotes, header, footer, algorithm, formula, formula number, image, table, seal, figure_table title, chart, and sidebar text and lists of references. The key metrics are as follow:

| Model| mAP(0.5) (%) | 
|  --- | --- | 
|PP-DocLayout_plus-L |  83.2 | 

**Note**: the evaluation set of the above precision indicators is the self built version sub area detection data set, including Chinese and English papers, magazines, newspapers, research reports PPT、 1000 document type pictures such as test papers and textbooks.


## Quick Start

### Installation

1. PaddlePaddle

Please refer to the following commands to install PaddlePaddle using pip:

```bash
# for CUDA11.8
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# for CUDA12.6
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# for CPU
python -m pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

For details about PaddlePaddle installation, please refer to the [PaddlePaddle official website](https://www.paddlepaddle.org.cn/en/install/quick).

2. PaddleOCR

Install the latest version of the PaddleOCR inference package from PyPI:

```bash
python -m pip install paddleocr
```


### Model Usage

You can quickly experience the functionality with a single command:

```bash
paddleocr layout_detection \
    --model_name PP-DocLayout_plus-L \
    -i https://cdn-uploads.huggingface.co/production/uploads/63d7b8ee07cd1aa3c49a2026/N5C68HPVAI-xQAWTxpbA6.jpeg
```

You can also integrate the model inference of the layout detection module into your project. Before running the following code, please download the sample image to your local machine.

```python
from paddleocr import LayoutDetection

model = LayoutDetection(model_name="PP-DocLayout_plus-L")
output = model.predict("N5C68HPVAI-xQAWTxpbA6.jpeg", batch_size=1, layout_nms=True)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
```

After running, the obtained result is as follows:

```json
{'res': {'input_path': '/root/.paddlex/predict_input/N5C68HPVAI-xQAWTxpbA6.jpeg', 'page_index': None, 'boxes': [{'cls_id': 2, 'label': 'text', 'score': 0.9870168566703796, 'coordinate': [34.101395, 349.85275, 358.5929, 611.0788]}, {'cls_id': 2, 'label': 'text', 'score': 0.986599326133728, 'coordinate': [34.500305, 647.15753, 358.29437, 848.66925]}, {'cls_id': 2, 'label': 'text', 'score': 0.984662652015686, 'coordinate': [385.71417, 497.41037, 711.22656, 697.8426]}, {'cls_id': 8, 'label': 'table', 'score': 0.9841272234916687, 'coordinate': [73.76732, 105.94854, 321.95355, 298.85074]}, {'cls_id': 8, 'label': 'table', 'score': 0.983431875705719, 'coordinate': [436.95523, 105.81446, 662.71814, 313.4865]}, {'cls_id': 2, 'label': 'text', 'score': 0.9832285642623901, 'coordinate': [385.62766, 346.22888, 710.10205, 458.772]}, {'cls_id': 2, 'label': 'text', 'score': 0.9816107749938965, 'coordinate': [385.78085, 735.19293, 710.5613, 849.97656]}, {'cls_id': 6, 'label': 'figure_title', 'score': 0.9577467441558838, 'coordinate': [34.421764, 20.055021, 358.7124, 76.53721]}, {'cls_id': 6, 'label': 'figure_title', 'score': 0.9505674839019775, 'coordinate': [385.7235, 20.054104, 711.2928, 74.92819]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.9001894593238831, 'coordinate': [386.46353, 477.035, 699.4023, 490.07495]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.8846081495285034, 'coordinate': [35.413055, 627.7365, 185.58315, 640.522]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.8837621808052063, 'coordinate': [387.1759, 716.34235, 524.78345, 729.2588]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.8509567975997925, 'coordinate': [35.50049, 331.18472, 141.64497, 344.81168]}]}}
```

The visualized image is as follows:

![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/63d7b8ee07cd1aa3c49a2026/5gAq1cFy1IX_wX26C2XmM.jpeg)

For details about usage command and descriptions of parameters, please refer to the [Document](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/module_usage/layout_detection.html#iii-quick-integration).

### Pipeline Usage

The ability of a single model is limited. But the pipeline consists of several models can provide more capacity to resolve difficult problems in real-world scenarios.

#### PP-StructureV3

Layout analysis is a technique used to extract structured information from document images. PP-StructureV3 includes the following six modules:
* Layout Detection Module
* General OCR Sub-pipeline
* Document Image Preprocessing Sub-pipeline （Optional）
* Table Recognition Sub-pipeline （Optional）
* Seal Recognition Sub-pipeline （Optional）
* Formula Recognition Sub-pipeline （Optional）

You can quickly experience the PP-StructureV3 pipeline with a single command.

```bash
paddleocr pp_structurev3 -i https://cdn-uploads.huggingface.co/production/uploads/63d7b8ee07cd1aa3c49a2026/KP10tiSZfAjMuwZUSLtRp.png
```

You can experience the inference of the pipeline with just a few lines of code. Taking the PP-StructureV3 pipeline as an example:

```python
from paddleocr import PPStructureV3

pipeline = PPStructureV3()
# ocr = PPStructureV3(use_doc_orientation_classify=True) # Use use_doc_orientation_classify to enable/disable document orientation classification model
# ocr = PPStructureV3(use_doc_unwarping=True) # Use use_doc_unwarping to enable/disable document unwarping module
# ocr = PPStructureV3(use_textline_orientation=True) # Use use_textline_orientation to enable/disable textline orientation classification model
# ocr = PPStructureV3(device="gpu") # Use device to specify GPU for model inference
output = pipeline.predict("./KP10tiSZfAjMuwZUSLtRp.png")
for res in output:
    res.print() ## Print the structured prediction output
    res.save_to_json(save_path="output") ## Save the current image's structured result in JSON format
    res.save_to_markdown(save_path="output") ## Save the current image's result in Markdown format
```

The default model used in pipeline is `PP-DocLayout_plus-L`.
For details about usage command and descriptions of parameters, please refer to the [Document](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/PP-StructureV3.html#2-quick-start).

## Links

[PaddleOCR Repo](https://github.com/paddlepaddle/paddleocr)

[PaddleOCR Documentation](https://paddlepaddle.github.io/PaddleOCR/latest/en/index.html)

