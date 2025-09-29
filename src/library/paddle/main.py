import os
import json
from paddleocr import LayoutDetection, TableClassification, TableCellsDetection, TextDetection
from PIL import Image as pillowImage

os.makedirs("/home/app/file/output/paddle/layout/", exist_ok=True)
os.makedirs("/home/app/file/output/paddle/table/crop/", exist_ok=True)
os.makedirs("/home/app/file/output/paddle/table/classification/", exist_ok=True)
os.makedirs("/home/app/file/output/paddle/table/wired/", exist_ok=True)
os.makedirs("/home/app/file/output/paddle/table/wireless/", exist_ok=True)
os.makedirs("/home/app/file/output/paddle/export/", exist_ok=True)

pathModelLayout = "/home/app/src/library/paddle/PP-DocLayout_plus-L/"
pathModelTableClassification = "/home/app/src/library/paddle/PP-LCNet_x1_0_table_cls/"
pathModelTableWired = "/home/app/src/library/paddle/RT-DETR-L_wired_table_cell_det/"
pathModelTableWireless = "/home/app/src/library/paddle/RT-DETR-L_wireless_table_cell_det/"
pathModelTextDetection = "/home/app/src/library/paddle/PP-OCRv5_server_det/"
pathImageInput = "/home/app/file/input/1_jp.jpg"
pathLayout = "/home/app/file/output/paddle/layout/"
pathTable = "/home/app/file/output/paddle/table/"
pathExport = "/home/app/file/output/paddle/export/"

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
        outputTextDetection.save_to_img(save_path=f"{pathExport}{fileNameSplit}.jpg")
        outputTextDetection.save_to_json(save_path=f"{pathExport}{fileNameSplit}.json")

def process():
    imageInput = pillowImage.open(pathImageInput)

    return imageInput

def inferenceLayout():
    modelLayout = LayoutDetection(model_dir=pathModelLayout, model_name="PP-DocLayout_plus-L")
    outputBlockLayoutList = modelLayout.predict(input=pathImageInput, batch_size=1)

    for outputBlockLayout in outputBlockLayoutList:
        outputBlockLayout.save_to_img(save_path=f"{pathLayout}result.jpg")
        outputBlockLayout.save_to_json(save_path=f"{pathLayout}result.json")

def readLabelTable(imageInput):
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

        _inferenceTableClassification(f"{pathTable}crop/{count}.jpg")
                                    
        count += 1

def readLabelTableCell():
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
            resultScore = scoreList[resultIndex]

            if (resultLabel == "wired_table"):
                _inferenceTableWired(inputPath)
            elif (resultLabel == "wireless_table"):
                _inferenceTableWireless(inputPath)

def exportText():
    for file in os.listdir(f"{pathTable}crop/"):
        _inferenceTextDetection(f"{pathTable}crop/{file}")

imageInput = process()

inferenceLayout()

readLabelTable(imageInput)

readLabelTableCell()

exportText()