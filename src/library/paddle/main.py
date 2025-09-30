import os
import json
import pandas
import cv2
from paddleocr import LayoutDetection, TableClassification, TableCellsDetection, TextDetection, TextRecognition, PaddleOCR, TableRecognitionPipelineV2
from PIL import Image as pillowImage

os.makedirs("/home/app/file/output/paddle/layout/", exist_ok=True)
os.makedirs("/home/app/file/output/paddle/table/crop/", exist_ok=True)
os.makedirs("/home/app/file/output/paddle/table/classification/", exist_ok=True)
os.makedirs("/home/app/file/output/paddle/table/wired/", exist_ok=True)
os.makedirs("/home/app/file/output/paddle/table/wireless/", exist_ok=True)
os.makedirs("/home/app/file/output/paddle/export/detection/", exist_ok=True)
os.makedirs("/home/app/file/output/paddle/text/crop/", exist_ok=True)
os.makedirs("/home/app/file/output/paddle/export/recognition/", exist_ok=True)

pathModelLayout = "/home/app/src/library/paddle/PP-DocLayout_plus-L/"
pathModelTableClassification = "/home/app/src/library/paddle/PP-LCNet_x1_0_table_cls/"
pathModelTableWired = "/home/app/src/library/paddle/RT-DETR-L_wired_table_cell_det/"
pathModelTableWireless = "/home/app/src/library/paddle/RT-DETR-L_wireless_table_cell_det/"
pathModelTextDetection = "/home/app/src/library/paddle/PP-OCRv5_server_det/"
pathModelTextRecognition = "/home/app/src/library/paddle/PP-OCRv5_server_rec/"

pathImageInput = "/home/app/file/input/1_jp.jpg"
pathLayout = "/home/app/file/output/paddle/layout/"
pathTable = "/home/app/file/output/paddle/table/"
pathExport = "/home/app/file/output/paddle/export/"
pathText = "/home/app/file/output/paddle/text/"

isTableClassification = False
isTableOcr=True

def _inferenceTableClassification(path):
    modelTableClassification = TableClassification(model_dir=pathModelTableClassification, model_name="PP-LCNet_x1_0_table_cls")
    outputTableClassificationList = modelTableClassification.predict(input=path, batch_size=1)

    fileNameSplit, _ = os.path.splitext(os.path.basename(path))

    for outputTableClassification in outputTableClassificationList:
        outputTableClassification.save_to_img(save_path=f"{pathTable}classification/{fileNameSplit}.jpg")
        outputTableClassification.save_to_json(save_path=f"{pathTable}classification/{fileNameSplit}.json")

def _inferenceTableWired(path):
    modelTableWired = TableCellsDetection(model_dir=pathModelTableWired, model_name="RT-DETR-L_wired_table_cell_det")
    outputTableWiredList = modelTableWired.predict(input=path, batch_size=1)

    fileNameSplit, _ = os.path.splitext(os.path.basename(path))

    for outputTableWired in outputTableWiredList:
        outputTableWired.save_to_img(save_path=f"{pathTable}wired/{fileNameSplit}.jpg")
        outputTableWired.save_to_json(save_path=f"{pathTable}wired/{fileNameSplit}.json")

def _inferenceTableWireless(path):
    modelTableWireless = TableCellsDetection(model_dir=pathModelTableWireless, model_name="RT-DETR-L_wireless_table_cell_det")
    outputTableWirelessList = modelTableWireless.predict(input=path, batch_size=1)

    fileNameSplit, _ = os.path.splitext(os.path.basename(path))

    for outputTableWireless in outputTableWirelessList:
        outputTableWireless.save_to_img(save_path=f"{pathTable}wireless/{fileNameSplit}.jpg")
        outputTableWireless.save_to_json(save_path=f"{pathTable}wireless/{fileNameSplit}.json")

def _inferenceTextDetection(path):
    modelTextDetection = TextDetection(model_dir=pathModelTextDetection, model_name="PP-OCRv5_server_det")
    outputTextDetectionList = modelTextDetection.predict(input=path, batch_size=1)

    fileNameSplit, _ = os.path.splitext(os.path.basename(path))

    for outputTextDetection in outputTextDetectionList:
        outputTextDetection.save_to_img(save_path=f"{pathExport}detection/{fileNameSplit}.jpg")
        outputTextDetection.save_to_json(save_path=f"{pathExport}detection/{fileNameSplit}.json")

def _inferenceTextRecognition(path, countParent):
    modelTextRecognition = TextRecognition(model_dir=pathModelTextRecognition, model_name="PP-OCRv5_server_rec")
    outputTextRecognitionList = modelTextRecognition.predict(input=path, batch_size=1)

    fileNameSplit, _ = os.path.splitext(os.path.basename(path))

    os.makedirs(f"{pathExport}recognition/{countParent}/", exist_ok=True)

    for outputTextRecognition in outputTextRecognitionList:
        outputTextRecognition.save_to_img(save_path=f"{pathExport}recognition/{countParent}/{fileNameSplit}.jpg")
        outputTextRecognition.save_to_json(save_path=f"{pathExport}recognition/{countParent}/{fileNameSplit}.json")

def _jsonToExcel(jsonPath, outputExcelPath):
    with open(jsonPath, "r", encoding="utf-8") as file:
        data = json.load(file)

    texts = data.get("rec_texts", [])
    boxes = data.get("rec_boxes", [])

    items = []
    for a in range(len(texts)):
        box = boxes[a]
        items.append({
            "text": texts[a],
            "x": box[0],
            "y": box[1],
            "w": box[2] - box[0],
            "h": box[3] - box[1]
        })

    items.sort(key=lambda item: (item["y"], item["x"]))

    rows = []
    for b in range(len(items)):
        item = items[b]
        placed = False
        for c in range(len(rows)):
            row = rows[c]
            if abs(row[0]["y"] - item["y"]) < max(row[0]["h"], item["h"]) / 2:
                row.append(item)
                placed = True
                break
        if not placed:
            rows.append([item])

    columnPositions = []
    for d in range(len(items)):
        x = items[d]["x"]
        matched = False
        for e in range(len(columnPositions)):
            if abs(x - columnPositions[e]) < 10:
                matched = True
                break
        if not matched:
            columnPositions.append(x)
    columnPositions.sort()

    table = []
    for f in range(len(rows)):
        rowItems = rows[f]
        row = ["" for _ in columnPositions]
        for g in range(len(rowItems)):
            item = rowItems[g]
            for h in range(len(columnPositions)):
                if abs(item["x"] - columnPositions[h]) < 10:
                    row[h] = item["text"]
                    break
        table.append(row)

    df = pandas.DataFrame(table)
    df.to_excel(outputExcelPath, index=False, header=False)

def _inferenceOcr(path):
    ocr = PaddleOCR(
        text_detection_model_dir=pathModelTextDetection,
        text_detection_model_name="PP-OCRv5_server_det",
        text_recognition_model_dir=pathModelTextRecognition,
        text_recognition_model_name="PP-OCRv5_server_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu"
    )

    resultList = ocr.predict(input=f"{path}")

    fileNameSplit, _ = os.path.splitext(os.path.basename(f"{path}"))

    for result in resultList:
        result.save_to_img(save_path=f"{pathExport}recognition/{fileNameSplit}.jpg")
        result.save_to_json(save_path=f"{pathExport}recognition/{fileNameSplit}.json")
        
        #_jsonToExcel(f"{pathExport}recognition/{fileNameSplit}.json", f"{pathExport}recognition/{fileNameSplit}.xlsx")

def process():
    imageInput = pillowImage.open(pathImageInput)

    return imageInput

def inferenceLayout():
    modelLayout = LayoutDetection(model_dir=pathModelLayout, model_name="PP-DocLayout_plus-L")
    outputBlockLayoutList = modelLayout.predict(input=pathImageInput, batch_size=1)

    for outputBlockLayout in outputBlockLayoutList:
        outputBlockLayout.save_to_img(save_path=f"{pathLayout}result.jpg")
        outputBlockLayout.save_to_json(save_path=f"{pathLayout}result.json")

def readTable(imageInput, isClassification=False, isOcr=False):
    with open(f"{pathLayout}result.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    boxList = data.get("boxes", [])

    count = 0

    for box in boxList:
        if str(box.get("label", "")).lower() != "table":
            continue

        coordinateList = box.get("coordinate", [])

        x1, y1, x2, y2 = coordinateList

        imageCrop = imageInput.crop((x1, y1, x2, y2))
        imageCrop.save(f"{pathTable}crop/{count}.jpg", quality=100)

        if isClassification:
            _inferenceTableClassification(f"{pathTable}crop/{count}.jpg")

        if isOcr:
            _inferenceOcr(f"{pathTable}crop/{count}.jpg")
                                    
        count += 1

def readTableCell():
    jsonFileList = []

    for file in os.listdir(f"{pathTable}classification/"):
        if file.lower().endswith(".json"):
            jsonFileList.append(file)

    for jsonFile in jsonFileList:
        with open(f"{pathTable}classification/{jsonFile}", "r", encoding="utf-8") as file:
            data = json.load(file)
        
        inputPath = data.get("input_path", "")
        labelNameList = data.get("label_names", [])
        scoreList = data.get("scores", [])

        if len(labelNameList) == len(scoreList):
            resultIndex = scoreList.index(max(scoreList))

            resultLabel = labelNameList[resultIndex]

            if (resultLabel == "wired_table"):
                _inferenceTableWired(inputPath)
            elif (resultLabel == "wireless_table"):
                _inferenceTableWireless(inputPath)

def textDetection():
    for file in os.listdir(f"{pathTable}crop/"):
        _inferenceTextDetection(f"{pathTable}crop/{file}")

def textRecognition():
    jsonFileList = []

    for file in os.listdir(f"{pathExport}detection/"):
        if file.lower().endswith(".json"):
            jsonFileList.append(file)
    
    countParent = 0
    
    for jsonFile in jsonFileList:
        countChildren = 0

        with open(f"{pathExport}detection/{jsonFile}", "r", encoding="utf-8") as file:
            data = json.load(file)

        inputPath = data.get("input_path", "")
        polygonList = data.get("dt_polys", [])

        imageInput = pillowImage.open(inputPath)

        for polygon in polygonList:
            xList = [point[0] for point in polygon]
            yList = [point[1] for point in polygon]
            x1, y1, x2, y2 = min(xList), min(yList), max(xList), max(yList)

            os.makedirs(f"{pathText}crop/{countParent}/", exist_ok=True)

            imageCrop = imageInput.crop((x1, y1, x2, y2))
            imageCrop.save(f"{pathText}crop/{countParent}/{countChildren}.jpg", quality=100)

            _inferenceTextRecognition(f"{pathText}crop/{countParent}/{countChildren}.jpg", countParent)

            countChildren += 1
        
        countParent += 1

#imageInput = process()

#inferenceLayout()

#readTable(imageInput, isTableClassification, isTableOcr)

#if isTableClassification:
#    readTableCell()

#textDetection()

#textRecognition()

#_jsonToExcel(f"{pathExport}recognition/1.json", f"{pathExport}recognition/1.xlsx")

pipeline = TableRecognitionPipelineV2(
    layout_detection_model_dir=pathModelLayout,
    layout_detection_model_name="PP-DocLayout_plus-L",
    table_classification_model_dir=pathModelTableClassification,
    table_classification_model_name="PP-LCNet_x1_0_table_cls",
    wired_table_cells_detection_model_dir=pathModelTableWired,
    wired_table_cells_detection_model_name="RT-DETR-L_wired_table_cell_det",
    wireless_table_cells_detection_model_dir=pathModelTableWireless,
    wireless_table_cells_detection_model_name="RT-DETR-L_wireless_table_cell_det",
    text_detection_model_dir=pathModelTextDetection,
    text_detection_model_name="PP-OCRv5_server_det",
    text_recognition_model_dir=pathModelTextRecognition,
    text_recognition_model_name="PP-OCRv5_server_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    device="cpu"
)
output = pipeline.predict(f"{pathTable}crop/1.jpg")

for res in output:
    res.save_to_xlsx(pathExport)