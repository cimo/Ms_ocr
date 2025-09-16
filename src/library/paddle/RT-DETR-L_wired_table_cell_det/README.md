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
- table_cells_detection
---

# RT-DETR-L_wired_table_cell_det

## Introduction

The Table Cell Detection Module is a key component of the table recognition task, responsible for locating and marking each cell region in table images. The performance of this module directly affects the accuracy and efficiency of the entire table recognition process. The Table Cell Detection Module typically outputs bounding boxes for each cell region, which are then passed as input to the table recognition pipeline for further processing.


<table>
<tr>
<th>Model</th>
<th>Top1 Acc(%)</th>
<th>GPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
</tr>
<tr>
<td>RT-DETR-L_wired_table_cell_det</td>
<td>82.7</td>
<td>35.00 / 10.45</td>
<td>495.51 / 495.51</td>
<td>124M</td>
</tr>
</table>

**Note**: The accuracy of RT-DETR-L_wired_table_cell_det comes from the results of joint testing with RT-DETR-L_wireless_table_cell_det.


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
paddleocr table_cells_detection \
    --model_name RT-DETR-L_wired_table_cell_det \
    -i https://cdn-uploads.huggingface.co/production/uploads/681c1ecd9539bdde5ae1733c/JUU_5wJWVo4PcmJhSdIo3.png
```

You can also integrate the model inference of the table classification module into your project. Before running the following code, please download the sample image to your local machine.

```python
from paddleocr import TableCellsDetection
model = TableCellsDetection(model_name="RT-DETR-L_wired_table_cell_det")
output = model.predict("JUU_5wJWVo4PcmJhSdIo3.png", threshold=0.3, batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```

After running, the obtained result is as follows:

```json
{'res': {'input_path': 'JUU_5wJWVo4PcmJhSdIo3.png', 'page_index': None, 'boxes': [{'cls_id': 0, 'label': 'cell', 'score': 0.9719462394714355, 'coordinate': [98.776054, 48.676155, 235.74197, 94.76812]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9706293344497681, 'coordinate': [235.65723, 48.66303, 473.31378, 94.746185]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9692592620849609, 'coordinate': [235.62718, 164.7009, 473.3329, 211.70175]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9682302474975586, 'coordinate': [98.61444, 164.80591, 235.63733, 211.60106]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9662815928459167, 'coordinate': [1.914098, 48.64288, 98.82235, 94.75366]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9643649458885193, 'coordinate': [1.8260963, 164.74123, 98.64024, 211.56848]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9605159759521484, 'coordinate': [98.783226, 117.873886, 235.74089, 141.91118]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9604074358940125, 'coordinate': [98.77425, 94.79676, 235.80171, 117.937065]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9603073596954346, 'coordinate': [98.788315, 1.8037335, 235.8512, 24.844206]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9592577815055847, 'coordinate': [235.70949, 94.7883, 473.3138, 117.90771]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9591122269630432, 'coordinate': [98.85015, 24.80603, 235.73082, 48.770897]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9586214423179626, 'coordinate': [235.62253, 1.8327671, 473.30493, 24.799725]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9583646059036255, 'coordinate': [235.7168, 117.81723, 473.26074, 141.87694]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9580551385879517, 'coordinate': [98.747986, 141.79, 235.71774, 164.90057]}, {'cls_id': 0, 'label': 'cell', 'score': 0.957258939743042, 'coordinate': [235.6782, 24.70515, 473.0595, 48.79732]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9568949937820435, 'coordinate': [1.8317447, 94.74939, 98.85935, 117.94785]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9563664793968201, 'coordinate': [1.8571337, 1.8207415, 98.98403, 24.901613]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9562588334083557, 'coordinate': [235.67096, 141.72911, 473.3746, 164.82388]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9557535648345947, 'coordinate': [1.922168, 117.84509, 98.85703, 141.85947]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9551460146903992, 'coordinate': [1.8364778, 141.7853, 98.83259, 164.88046]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9547295570373535, 'coordinate': [2.0152304, 24.793072, 98.84856, 48.75716]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9525823593139648, 'coordinate': [235.63931, 211.63988, 473.2472, 254.16182]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9454454779624939, 'coordinate': [98.62049, 211.4913, 235.57971, 254.40237]}, {'cls_id': 0, 'label': 'cell', 'score': 0.9410758018493652, 'coordinate': [1.9204835, 211.48651, 98.601524, 254.9897]}]}}
```

The visualized image is as follows:

![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/681c1ecd9539bdde5ae1733c/r81iuViiWrTCMnYaxidwv.png)

For details about usage command and descriptions of parameters, please refer to the [Document](https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/module_usage/table_cells_detection.html#iii-quick-start).

### Pipeline Usage

The ability of a single model is limited. But the pipeline consists of several models can provide more capacity to resolve difficult problems in real-world scenarios.

#### General Table Recognition V2 Pipeline

The general table recognition V2 pipeline is used to solve table recognition tasks by extracting information from images and outputting it in HTML or Excel format. And there are 8 modules in the pipeline: 
* Table Classification Module
* Table Structure Recognition Module
* Table Cell Detection Module
* Text Detection Module
* Text Recognition Module
* Layout Region Detection Module (Optional)
* Document Image Orientation Classification Module (Optional)
* Text Image Unwarping Module (Optional)

Run a single command to quickly experience the general table recognition V2 pipeline:

```bash

paddleocr table_recognition_v2 -i https://cdn-uploads.huggingface.co/production/uploads/681c1ecd9539bdde5ae1733c/mabagznApI1k9R8qFoTLc.png  \
    --use_doc_orientation_classify False  \
    --use_doc_unwarping False \
    --save_path ./output \
    --device gpu:0 
```

Results are printed to the terminal:

```json
{'res': {'input_path': 'mabagznApI1k9R8qFoTLc.png', 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_layout_detection': True, 'use_ocr_model': True}, 'layout_det_res': {'input_path': None, 'page_index': None, 'boxes': [{'cls_id': 8, 'label': 'table', 'score': 0.86655592918396, 'coordinate': [0.0125130415, 0.41920784, 1281.3737, 585.3884]}]}, 'overall_ocr_res': {'input_path': None, 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_textline_orientation': False}, 'dt_polys': array([[[   9,   21],
        ...,
        [   9,   59]],

       ...,

       [[1046,  536],
        ...,
        [1046,  573]]], dtype=int16), 'text_det_params': {'limit_side_len': 960, 'limit_type': 'max', 'thresh': 0.3, 'box_thresh': 0.6, 'unclip_ratio': 2.0}, 'text_type': 'general', 'textline_orientation_angles': array([-1, ..., -1]), 'text_rec_score_thresh': 0, 'rec_texts': ['部门', '报销人', '报销事由', '批准人：', '单据', '张', '合计金额', '元', '车费票', '其', '火车费票', '飞机票', '中', '旅住宿费', '其他', '补贴'], 'rec_scores': array([0.99958128, ..., 0.99317062]), 'rec_polys': array([[[   9,   21],
        ...,
        [   9,   59]],

       ...,

       [[1046,  536],
        ...,
        [1046,  573]]], dtype=int16), 'rec_boxes': array([[   9, ...,   59],
       ...,
       [1046, ...,  573]], dtype=int16)}, 'table_res_list': [{'cell_box_list': [array([ 0.13052222, ..., 73.08310249]), array([104.43082511, ...,  73.27777413]), array([319.39041221, ...,  73.30439308]), array([424.2436837 , ...,  73.44736794]), array([580.75836265, ...,  73.24003914]), array([723.04370201, ...,  73.22717598]), array([984.67315757, ...,  73.20420387]), array([1.25130415e-02, ..., 5.85419208e+02]), array([984.37072837, ..., 137.02281502]), array([984.26586998, ..., 201.22290352]), array([984.24017417, ..., 585.30775765]), array([1039.90606773, ...,  265.44664314]), array([1039.69549644, ...,  329.30540779]), array([1039.66546714, ...,  393.57319954]), array([1039.5122689 , ...,  457.74644783]), array([1039.55535972, ...,  521.73030403]), array([1039.58612144, ...,  585.09468392])], 'pred_html': '<html><body><table><tbody><tr><td>部门</td><td></td><td>报销人</td><td></td><td>报销事由</td><td></td><td colspan="2">批准人：</td></tr><tr><td colspan="6" rowspan="8"></td><td colspan="2">单据 张</td></tr><tr><td colspan="2">合计金额 元</td></tr><tr><td rowspan="6">其 中</td><td>车费票</td></tr><tr><td>火车费票</td></tr><tr><td>飞机票</td></tr><tr><td>旅住宿费</td></tr><tr><td>其他</td></tr><tr><td>补贴</td></tr></tbody></table></body></html>', 'table_ocr_pred': {'rec_polys': array([[[   9,   21],
        ...,
        [   9,   59]],

       ...,

       [[1046,  536],
        ...,
        [1046,  573]]], dtype=int16), 'rec_texts': ['部门', '报销人', '报销事由', '批准人：', '单据', '张', '合计金额', '元', '车费票', '其', '火车费票', '飞机票', '中', '旅住宿费', '其他', '补贴'], 'rec_scores': array([0.99958128, ..., 0.99317062]), 'rec_boxes': array([[   9, ...,   59],
       ...,
       [1046, ...,  573]], dtype=int16)}}]}}
```

If save_path is specified, the visualization results will be saved under `save_path`. The visualization output is shown below:

![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/681c1ecd9539bdde5ae1733c/b3mPpaMsK049qxsTbotvI.png)

The command-line method is for quick experience. For project integration, also only a few codes are needed as well:

```python
from paddleocr import TableRecognitionPipelineV2

pipeline = TableRecognitionPipelineV2(
    use_doc_orientation_classify=False, # Use use_doc_orientation_classify to enable/disable document orientation classification model
    use_doc_unwarping=False, # Use use_doc_unwarping to enable/disable document unwarping module
)
# pipeline = TableRecognitionPipelineV2(use_doc_orientation_classify=True) # Specify whether to use the document orientation classification model with use_doc_orientation_classify
# pipeline = TableRecognitionPipelineV2(use_doc_unwarping=True) # Specify whether to use the text image unwarping module with use_doc_unwarping
# pipeline = TableRecognitionPipelineV2(device="gpu") # Specify the device to use GPU for model inference
output = pipeline.predict("https://cdn-uploads.huggingface.co/production/uploads/681c1ecd9539bdde5ae1733c/mabagznApI1k9R8qFoTLc.png")
for res in output:
    res.print() ## Print the predicted structured output
    res.save_to_img("./output/")
    res.save_to_xlsx("./output/")
    res.save_to_html("./output/")
    res.save_to_json("./output/")
```

For details about usage command and descriptions of parameters, please refer to the [Document](https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/pipeline_usage/table_recognition_v2.html#2-quick-start).

#### PP-StructureV3

Layout analysis is a technique used to extract structured information from document images. PP-StructureV3 includes the following six modules:
* Layout Detection Module
* General OCR Pipeline
* Document Image Preprocessing Pipeline （Optional）
* Table Recognition Pipeline （Optional）
* Seal Recognition Pipeline （Optional）
* Formula Recognition Pipeline （Optional）

Run a single command to quickly experience the PP-StructureV3 pipeline:

```bash
paddleocr pp_structurev3 -i https://cdn-uploads.huggingface.co/production/uploads/681c1ecd9539bdde5ae1733c/mG4tnwfrvECoFMu-S9mxo.png \
    --use_doc_orientation_classify False \
    --use_doc_unwarping False \
    --use_textline_orientation False \
    --device gpu:0
```

Results would be printed to the terminal. If save_path is specified, the results will be saved under `save_path`. 

Just a few lines of code can experience the inference of the pipeline. Taking the PP-StructureV3 pipeline as an example:

```python
from paddleocr import PPStructureV3

pipeline = PPStructureV3(
    use_doc_orientation_classify=False, # Use use_doc_orientation_classify to enable/disable document orientation classification model
    use_doc_unwarping=False,    # Use use_doc_unwarping to enable/disable document unwarping module
    use_textline_orientation=False, # Use use_textline_orientation to enable/disable textline orientation classification model
    device="gpu:0", # Use device to specify GPU for model inference
    )
output = pipeline.predict("SfxF0X4drBTNGnfFOtZij.png")
for res in output:
    res.print() # Print the structured prediction output
    res.save_to_json(save_path="output") ## Save the current image's structured result in JSON format
    res.save_to_markdown(save_path="output") ## Save the current image's result in Markdown format
```

For details about usage command and descriptions of parameters, please refer to the [Document](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/PP-StructureV3.html#2-quick-start).

## Links

[PaddleOCR Repo](https://github.com/paddlepaddle/paddleocr)

[PaddleOCR Documentation](https://paddlepaddle.github.io/PaddleOCR/latest/en/index.html)

