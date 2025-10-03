import os
import json
import numpy
import cv2
import openpyxl
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment
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

pathLayout = "/home/app/file/output/paddle/layout/"
pathTable = "/home/app/file/output/paddle/table/"
pathExport = "/home/app/file/output/paddle/export/"

isDebug = True
device = "cpu"

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

def _inferenceTableClassification(input, count):
    modelTableClassification = TableClassification(model_dir=pathModelTableClassification, model_name="PP-LCNet_x1_0_table_cls", device=device)
    outputTableClassificationList = modelTableClassification.predict(input=input, batch_size=1)

    for outputTableClassification in outputTableClassificationList:
        if isDebug:
            outputTableClassification.save_to_img(save_path=f"{pathTable}classification/{count}_debug.jpg")
        
        outputTableClassification.save_to_json(save_path=f"{pathTable}classification/{count}.json")

def _inferenceTableWired(input, count):
    modelTableWired = TableCellsDetection(model_dir=pathModelTableWired, model_name="RT-DETR-L_wired_table_cell_det", device=device)
    outputTableWiredList = modelTableWired.predict(input=input, batch_size=1)

    for outputTableWired in outputTableWiredList:
        if isDebug:
            outputTableWired.save_to_img(save_path=f"{pathTable}wired/{count}_debug.jpg")

        #outputTableWired.save_to_json(save_path=f"{pathTable}wired/{count}.json")
        print("cimo", outputTableWired)

def _inferenceTableWireless(input, count):
    modelTableWireless = TableCellsDetection(model_dir=pathModelTableWireless, model_name="RT-DETR-L_wireless_table_cell_det", device=device)
    outputTableWirelessList = modelTableWireless.predict(input=input, batch_size=1)

    for outputTableWireless in outputTableWirelessList:
        if isDebug:
            outputTableWireless.save_to_img(save_path=f"{pathTable}wireless/{count}_debug.jpg")

        outputTableWireless.save_to_json(save_path=f"{pathTable}wireless/{count}.json")

def _extractTableCell(pathJson, imageCrop, count):
    with open(pathJson, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    labelNameList = data.get("label_names", [])
    scoreList = data.get("scores", [])

    if len(labelNameList) == len(scoreList):
        resultIndex = scoreList.index(max(scoreList))
        resultLabel = labelNameList[resultIndex]

        if (resultLabel == "wired_table"):
            cv2.imwrite(f"{pathTable}wired/{count}.jpg", imageCrop)

            _inferenceTableWired(imageCrop, count)
        elif (resultLabel == "wireless_table"):
            cv2.imwrite(f"{pathTable}wireless/{count}.jpg", imageCrop)

            _inferenceTableWireless(imageCrop, count)

def _jsonToExcel(pathJson, pathExcel):
    with open(pathJson, "r", encoding="utf-8") as file:
        data = json.load(file)

    workbook = openpyxl.Workbook()
    sheet = workbook.active

    cells = []
    for item in data:
        recTexts = item.get("rec_texts", [])
        recPolys = item.get("rec_polys", [])
        for a in range(len(recTexts)):
            poly = recPolys[a]
            minX = min([point[0] for point in poly])
            minY = min([point[1] for point in poly])
            maxX = max([point[0] for point in poly])
            maxY = max([point[1] for point in poly])
            centerX = (minX + maxX) / 2
            centerY = (minY + maxY) / 2
            cells.append({
                "text": recTexts[a],
                "centerX": centerX,
                "centerY": centerY
            })

    # Ordina per posizione fisica
    cells.sort(key=lambda x: (x["centerY"], x["centerX"]))

    # Mappa logica: assegna riga e colonna
    rowCenters = []
    rowIndexMap = {}
    colIndexMap = {}
    layoutMap = []
    rowCounter = 1

    for cell in cells:
        y = cell["centerY"]
        x = cell["centerX"]

        # Assegna riga
        matchedRow = None
        for ry in rowCenters:
            if abs(ry - y) < 1e-3:
                matchedRow = rowIndexMap[ry]
                break
        if matchedRow is None:
            rowCenters.append(y)
            rowIndexMap[y] = rowCounter
            matchedRow = rowCounter
            rowCounter += 1
            colIndexMap[matchedRow] = []

        # Assegna colonna
        matchedCol = None
        for col in colIndexMap[matchedRow]:
            if abs(col["x"] - x) < 1e-3:
                matchedCol = col["index"]
                break
        if matchedCol is None:
            matchedCol = len(colIndexMap[matchedRow]) + 1
            colIndexMap[matchedRow].append({"x": x, "index": matchedCol})

        layoutMap.append({
            "row": matchedRow,
            "col": matchedCol,
            "text": cell["text"]
        })

    # Scrivi in Excel
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    for cell in layoutMap:
        sheet.cell(row=cell["row"], column=cell["col"]).value = cell["text"]

    workbook.save(pathExcel)

def process(pathImage):
    imageInput = pillowImage.open(pathImage)
    imageInput = numpy.array(imageInput)

    height, width = imageInput.shape[:2]
    maxSide = max(height, width)

    if maxSide > 960:
        scale = 960 / maxSide
        newWidth = int(width * scale)
        newHeight = int(height * scale)
        imageInput = cv2.resize(imageInput, (newWidth, newHeight))

    return imageInput

def inferenceLayout(input):
    modelLayout = LayoutDetection(model_dir=pathModelLayout, model_name="PP-DocLayout_plus-L", device=device)
    outputBlockLayoutList = modelLayout.predict(input=input, batch_size=1)

    for outputBlockLayout in outputBlockLayoutList:
        if isDebug:
            outputBlockLayout.save_to_img(save_path=f"{pathLayout}result_debug.jpg")
        
        outputBlockLayout.save_to_json(save_path=f"{pathLayout}result.json")

def extractTable(imageInput, isClassification=False):
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
            _extractTableCell(f"{pathTable}classification/{count}.json", imageCrop, count)
                                    
        count += 1

def processTable(pathJson, imageInput):
    resultMergeList = []

    with open(pathJson, "r", encoding="utf-8") as file:
        data = json.load(file)

    boxList = data.get("boxes", [])

    for box in boxList:
        coordinateList = box.get("coordinate", [])
        x1, y1, x2, y2 = map(int, coordinateList)

        imageCrop = imageInput[y1:y2, x1:x2, :]

        resultList = ocr.predict(input=imageCrop)

        for result in resultList:
            recPolyList = result.get("rec_polys", [])
            
            if isinstance(recPolyList, numpy.ndarray):
                recPolyList = recPolyList.tolist()
            else:
                for a in range(len(recPolyList)):
                    if isinstance(recPolyList[a], numpy.ndarray):
                        recPolyList[a] = recPolyList[a].tolist()

            resultMergeList.append({
                "rec_texts": result.get("rec_texts", []),
                "rec_scores": result.get("rec_scores", []),
                "rec_polys": recPolyList
            })
    
    with open(f"{pathExport}result.json", "w", encoding="utf-8") as file:
        json.dump(resultMergeList, file, ensure_ascii=False, indent=2)

    _jsonToExcel(f"{pathExport}result.json", f"{pathExport}result.xlsx")

imageInput = process("/home/app/file/input/1_jp.jpg")

inferenceLayout(imageInput)

extractTable(imageInput, True)

#inferenceOcr(imageInput)

#imageInput = process(f"{pathTable}wired/0.jpg")

#processTable(f"{pathTable}wired/0.json", imageInput)

#_jsonToExcel(f"{pathTable}wired/0.json", f"{pathExport}result.xlsx")