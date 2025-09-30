import os
import json
import pandas
import numpy
from paddleocr import LayoutDetection, TableClassification, TableCellsDetection, PaddleOCR
from PIL import Image as pillowImage

os.makedirs("/home/app/file/output/paddle/layout/", exist_ok=True)
os.makedirs("/home/app/file/output/paddle/table/classification/", exist_ok=True)
os.makedirs("/home/app/file/output/paddle/table/wired/", exist_ok=True)
os.makedirs("/home/app/file/output/paddle/table/wireless/", exist_ok=True)
os.makedirs("/home/app/file/output/paddle/export/", exist_ok=True)

pathModelLayout = "/home/app/src/library/paddle/PP-DocLayout_plus-L/"
pathModelTableClassification = "/home/app/src/library/paddle/PP-LCNet_x1_0_table_cls/"
pathModelTableWired = "/home/app/src/library/paddle/RT-DETR-L_wired_table_cell_det/"
pathModelTableWireless = "/home/app/src/library/paddle/RT-DETR-L_wireless_table_cell_det/"
pathModelTextDetection = "/home/app/src/library/paddle/PP-OCRv5_server_det/"
pathModelTextRecognition = "/home/app/src/library/paddle/PP-OCRv5_server_rec/"
pathModelChart = "/home/app/src/library/paddle/PP-Chart2Table/"

pathImageInput = "/home/app/file/input/Chart.jpg"
pathLayout = "/home/app/file/output/paddle/layout/"
pathTable = "/home/app/file/output/paddle/table/"
pathExport = "/home/app/file/output/paddle/export/"

isDebug = True
device = "cpu"

def _inferenceTableClassification(input, count):
    modelTableClassification = TableClassification(model_dir=pathModelTableClassification, model_name="PP-LCNet_x1_0_table_cls", device=device)
    outputTableClassificationList = modelTableClassification.predict(input=input, batch_size=1)

    for outputTableClassification in outputTableClassificationList:
        if isDebug:
            outputTableClassification.save_to_img(save_path=f"{pathTable}classification/{count}.jpg")
        
        outputTableClassification.save_to_json(save_path=f"{pathTable}classification/{count}.json")

def _inferenceTableWired(input, count):
    modelTableWired = TableCellsDetection(model_dir=pathModelTableWired, model_name="RT-DETR-L_wired_table_cell_det", device=device)
    outputTableWiredList = modelTableWired.predict(input=input, batch_size=1)

    for outputTableWired in outputTableWiredList:
        if isDebug:
            outputTableWired.save_to_img(save_path=f"{pathTable}wired/{count}.jpg")

        outputTableWired.save_to_json(save_path=f"{pathTable}wired/{count}.json")

def _inferenceTableWireless(input, count):
    modelTableWireless = TableCellsDetection(model_dir=pathModelTableWireless, model_name="RT-DETR-L_wireless_table_cell_det", device=device)
    outputTableWirelessList = modelTableWireless.predict(input=input, batch_size=1)

    for outputTableWireless in outputTableWirelessList:
        if isDebug:
            outputTableWireless.save_to_img(save_path=f"{pathTable}wireless/{count}.jpg")

        outputTableWireless.save_to_json(save_path=f"{pathTable}wireless/{count}.json")

def _readTableCell(pathJson, imageCrop, count):
    with open(pathJson, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    labelNameList = data.get("label_names", [])
    scoreList = data.get("scores", [])

    if len(labelNameList) == len(scoreList):
        resultIndex = scoreList.index(max(scoreList))
        resultLabel = labelNameList[resultIndex]

        if (resultLabel == "wired_table"):
            _inferenceTableWired(imageCrop, count)
        elif (resultLabel == "wireless_table"):
            _inferenceTableWireless(imageCrop, count)

def process():
    imageInput = pillowImage.open(pathImageInput)
    imageInput = numpy.array(imageInput)

    return imageInput

def inferenceLayout(input):
    modelLayout = LayoutDetection(model_dir=pathModelLayout, model_name="PP-DocLayout_plus-L", device=device)
    outputBlockLayoutList = modelLayout.predict(input=input, batch_size=1)

    for outputBlockLayout in outputBlockLayoutList:
        if isDebug:
            outputBlockLayout.save_to_img(save_path=f"{pathLayout}result.jpg")
        
        outputBlockLayout.save_to_json(save_path=f"{pathLayout}result.json")

def readTable(imageInput, isClassification=False):
    with open(f"{pathLayout}result.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    boxList = data.get("boxes", [])

    count = 0

    for box in boxList:
        if str(box.get("label", "")).lower() != "table":
            continue

        coordinateList = box.get("coordinate", [])
        x1, y1, x2, y2 = map(int, coordinateList)

        imageCrop = imageInput[y1:y2, x1:x2, :]

        if isClassification:
            _inferenceTableClassification(imageCrop, count)
            _readTableCell(f"{pathTable}classification/{count}.json", imageCrop, count)
                                    
        count += 1

def inferenceOcr(input):
    ocr = PaddleOCR(
        text_detection_model_dir=pathModelTextDetection,
        text_detection_model_name="PP-OCRv5_server_det",
        text_recognition_model_dir=pathModelTextRecognition,
        text_recognition_model_name="PP-OCRv5_server_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device=device
    )
    resultList = ocr.predict(input=input)

    for result in resultList:
        if isDebug:
            result.save_to_img(save_path=f"{pathExport}result.jpg")

        result.save_to_json(save_path=f"{pathExport}result.json")

def jsonToExcel(pathJson, pathExcel):
    with open(pathJson, "r", encoding="utf-8") as file:
        data = json.load(file)

    lineList = data["result"].strip().split("\n")

    rowList = []

    for line in lineList:
        cellList = line.split("|")
        
        resultCellList = []

        for cell in cellList:
            resultCellList.append(cell.strip())

        rowList.append(resultCellList)

    header = rowList[0]

    resultRowList = []

    for a in range(1, len(rowList)):
        resultRowList.append(rowList[a])

    pandaDataFrame = pandas.DataFrame(resultRowList, columns=header)
    pandaDataFrame.to_excel(pathExcel, index=False)

imageInput = process()

#inferenceLayout(imageInput)

#readTable(imageInput, True)

inferenceOcr(imageInput)

#jsonToExcel(f"{pathChart}result.json", f"{pathChart}result.xlsx")