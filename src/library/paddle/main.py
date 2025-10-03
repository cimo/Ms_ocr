import os
import shutil
import json
import numpy
import cv2
from openpyxl import Workbook
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter
from paddleocr import LayoutDetection, TableClassification, TableCellsDetection, TextDetection, TextRecognition, PaddleOCR
from PIL import Image, ImageDraw, ImageFont

PATH_PADDLE_SYSTEM = "/home/app/.paddlex/"
PATH_PADDLE_FILE_OUTPUT = "/home/app/file/output/paddle/"
PATH_PADDLE_LIBRARY = "/home/app/src/library/paddle/"

pathFont = f"{PATH_PADDLE_SYSTEM}fonts/PingFang-SC-Regular.ttf"

pathModelLayout = f"{PATH_PADDLE_LIBRARY}PP-DocLayout_plus-L/"
pathModelTableClassification = f"{PATH_PADDLE_LIBRARY}PP-LCNet_x1_0_table_cls/"
pathModelTableWired = f"{PATH_PADDLE_LIBRARY}RT-DETR-L_wired_table_cell_det/"
pathModelTableWireless = f"{PATH_PADDLE_LIBRARY}RT-DETR-L_wireless_table_cell_det/"
pathModelTextDetection = f"{PATH_PADDLE_LIBRARY}PP-OCRv5_server_det/"
pathModelTextRecognition = f"{PATH_PADDLE_LIBRARY}PP-OCRv5_server_rec/"

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

def _inferenceTextDetection(input):
    model = TextDetection(model_dir=pathModelTextDetection, model_name="PP-OCRv5_server_det")
    outputList = model.predict(input=input, batch_size=1)

    for output in outputList:
        if isDebug:
            output.save_to_img(save_path=f"{PATH_PADDLE_FILE_OUTPUT}export/result_detection.jpg")
            output.save_to_json(save_path=f"{PATH_PADDLE_FILE_OUTPUT}export/result_detection.json")

def _inferenceTextRecognition(input):
    model = TextRecognition(model_dir=pathModelTextRecognition, model_name="PP-OCRv5_server_rec")
    outputList = model.predict(input=input, batch_size=1)

    for output in outputList:
        if isDebug:
            output.save_to_img(save_path=f"{PATH_PADDLE_FILE_OUTPUT}export/result_recognition.jpg")
            output.save_to_json(save_path=f"{PATH_PADDLE_FILE_OUTPUT}export/result_recognition.json")

def _filterOverlapBox(boxList):
    resultBoxFilterList = []

    for box in boxList:
        x1, y1, x2, y2 = map(float, box["coordinate"])
        
        isIgnore = False
        
        for boxFilter in resultBoxFilterList:
            xf1, yf1, xf2, yf2 = map(float, boxFilter["coordinate"])

            overlapX = max(0, min(x2, xf2) - max(x1, xf1))
            overlapY = max(0, min(y2, yf2) - max(y1, yf1))

            minWidth = min(x2 - x1, xf2 - xf1)
            minHeight = min(y2 - y1, yf2 - yf1)

            if overlapX / minWidth > 0.9 and overlapY / minHeight > 0.9:
                isIgnore = True

                break

        if isIgnore:
            continue

        resultBoxFilterList.append(box)
    
    return resultBoxFilterList
'''
def _jsonToExcel(data, fontSize, pathExcel):
    workbook = Workbook()
    sheet = workbook.active

    dataSortList = sorted(data, key=lambda x: (x["coordinate"][1], x["coordinate"][0]))

    rowEdgeList = set()

    for item in dataSortList:
        _, y1, _, y2 = item["coordinate"]

        rowEdgeList.add(round(y1, 1))
        rowEdgeList.add(round(y2, 1))

    rowEdgeList = sorted(rowEdgeList)

    rowObject = {}

    for a in range(len(rowEdgeList) - 1):
        rowObject[(rowEdgeList[a], rowEdgeList[a + 1])] = a + 1

    columnPositionList = []

    for dataSort in dataSortList:
        x1, _, _, _ = dataSort["coordinate"]

        if not any(abs(x1 - a) < 5 for a in columnPositionList):
            columnPositionList.append(x1)
    
    columnPositionList = sorted(columnPositionList)

    for dataSort in dataSortList:
        x1, y1, x2, y2 = dataSort["coordinate"]
        textList = dataSort["rec_texts"]

        columnIndex = None

        for key, value in enumerate(columnPositionList):
            if abs(x1 - value) < 5:
                columnIndex = key + 1

                break

        rowStart = None
        rowEnd = None

        for (ya, yb), rowNumber in rowObject.items():
            if abs(ya - y1) < 5:
                rowStart = rowNumber

            if abs(yb - y2) < 5:
                rowEnd = rowNumber

        if rowStart is None:
            for (ya, yb), rowNumber in rowObject.items():
                if ya <= y1 < yb:
                    rowStart = rowNumber

                    break

        if rowEnd is None:
            for (ya, yb), rowNumber in rowObject.items():
                if ya < y2 <= yb:
                    rowEnd = rowNumber

                    break

        if rowEnd is None:
            rowEnd = rowStart

        if rowEnd > rowStart:
            sheet.merge_cells(start_row=rowStart, start_column=columnIndex, end_row=rowEnd, end_column=columnIndex)

            if len(textList) > 1:
                totalLineList = rowEnd - rowStart + 1
                baseLineList = len(textList)

                if baseLineList < totalLineList:
                    extraSlot = totalLineList - baseLineList

                    lineList = []

                    for key, value in enumerate(textList):
                        lineList.append(value)

                        if key < baseLineList - 1:
                            lineList.append("\n")

                            if extraSlot > 0:
                                lineList.append("\n")

                                extraSlot -= 1

                    value = "".join(lineList)
                else:
                    value = "\n".join(textList)
            else:
                value = textList[0] if textList else ""
        else:
            value = textList[0] if len(textList) == 1 else "\n".join(textList)

        cell = sheet.cell(row=rowStart, column=columnIndex)
        cell.value = value
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

        #columnLetter = get_column_letter(columnIndex)
        #sheet.column_dimensions[columnLetter].width = (x2 - x1) / 7
        #sheet.row_dimensions[rowStart].height = (y2 - y1) * 0.75

    workbook.save(pathExcel)
'''
def _jsonToExcel(data, fontSize, pathExcel):
    workbook = Workbook()
    sheet = workbook.active

    # Ordina i dati
    dataSortList = sorted(data, key=lambda x: (x["coordinate"][1], x["coordinate"][0]))

    # Calcola bordi righe unici
    rowEdgeList = set()
    for item in dataSortList:
        _, y1, _, y2 = item["coordinate"]
        rowEdgeList.add(round(y1, 1))
        rowEdgeList.add(round(y2, 1))
    rowEdgeList = sorted(rowEdgeList)

    # Mappa coppie di bordi a numero riga
    rowObject = {}
    for a in range(len(rowEdgeList) - 1):
        rowObject[(rowEdgeList[a], rowEdgeList[a + 1])] = a + 1

    # Calcola posizioni colonne
    columnPositionList = []
    for item in dataSortList:
        x1, _, _, _ = item["coordinate"]
        if not any(abs(x1 - a) < 5 for a in columnPositionList):
            columnPositionList.append(x1)
    columnPositionList = sorted(columnPositionList)

    # Inserisci dati cella per cella
    for item in dataSortList:
        x1, y1, _, _ = item["coordinate"]
        textList = item["rec_texts"]

        # Trova colonna
        columnIndex = None
        for idx, value in enumerate(columnPositionList):
            if abs(x1 - value) < 5:
                columnIndex = idx + 1
                break

        # Trova riga
        rowStart = None
        for (ya, yb), rowNum in rowObject.items():
            if ya <= y1 < yb:
                rowStart = rowNum
                break
        if rowStart is None:
            rowStart = 1  # default se non trovato

        # Inserisci testo (tutte le righe separate da \n)
        value = "\n".join(textList) if textList else ""
        cell = sheet.cell(row=rowStart, column=columnIndex)
        cell.value = value
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

        # Imposta altezza riga proporzionale al fontSize
        sheet.row_dimensions[rowStart].height = fontSize * 1.2

    workbook.save(pathExcel)

def _processTable(mode, data, input, count):
    if isDebug:
        inputHeight, inputWidth = input.shape[:2]
        resultImage = Image.new("RGB", (inputWidth, inputHeight), (255, 255, 255))
        imageDraw = ImageDraw.Draw(resultImage)
        resultMergeList = []

    boxList = data.get("boxes", [])

    boxFilterList = _filterOverlapBox(boxList)

    for box in boxFilterList:
        coordinateList = box.get("coordinate", [])
        x1, y1, x2, y2 = map(int, coordinateList)

        imageCrop = input[y1:y2, x1:x2, :]

        resultList = ocr.predict(input=imageCrop)

        for result in resultList:
            coordinateList = [float(a) for a in coordinateList]
            textList = result.get("rec_texts", []) or [""]

            if isDebug:
                lineNumber = max(1, len(textList))
                cellHeight = y2 - y1
                fontSize = max(8, int(cellHeight * 0.6 / lineNumber))
                font = ImageFont.truetype(pathFont, fontSize)

                bboxeList = [imageDraw.textbbox((0, 0), value, font=font) for value in textList]
                textHeightList = [bbox[3] - bbox[1] for bbox in bboxeList]
                textWidthList = [bbox[2] - bbox[0] for bbox in bboxeList]

                totalTextHeight = sum(textHeightList)

                if lineNumber > 1:
                    extraSpace = (cellHeight - totalTextHeight) / (lineNumber + 1)
                else:
                    extraSpace = (cellHeight - totalTextHeight) / 2

                currentY = y1 + extraSpace

                for key, value in enumerate(textList):
                    textWidth = textWidthList[key]
                    textHeight = textHeightList[key]
                    textPositionX = x1 + (x2 - x1 - textWidth) // 2
                    textPositionY = int(currentY)

                    imageDraw.text((textPositionX, textPositionY), value, font=font, fill=(0, 0, 0))

                    currentY += textHeight + extraSpace

                resultMergeList.append({
                    "coordinate": coordinateList,
                    "rec_texts": textList
                })

    if isDebug:
        resultImage.save(f"{PATH_PADDLE_FILE_OUTPUT}table/{mode}/{count}_result.jpg", format="JPEG")

        with open(f"{PATH_PADDLE_FILE_OUTPUT}table/{mode}/{count}_result.json", "w", encoding="utf-8") as file:
            json.dump(resultMergeList, file, ensure_ascii=False, indent=2)

    _jsonToExcel(resultMergeList, fontSize, f"{PATH_PADDLE_FILE_OUTPUT}table/{mode}/{count}_result.xlsx")

def _inferenceTableWireless(input, count):
    model = TableCellsDetection(model_dir=pathModelTableWireless, model_name="RT-DETR-L_wireless_table_cell_det", device=device)
    dataList = model.predict(input=input, batch_size=1)

    for data in dataList:
        if isDebug:
            data.save_to_img(save_path=f"{PATH_PADDLE_FILE_OUTPUT}table/wireless/{count}.jpg")
            data.save_to_json(save_path=f"{PATH_PADDLE_FILE_OUTPUT}table/wireless/{count}.json")

        _processTable("wireless", data, input, count)

def _inferenceTableWired(input, count):
    model = TableCellsDetection(model_dir=pathModelTableWired, model_name="RT-DETR-L_wired_table_cell_det", device=device)
    dataList = model.predict(input=input, batch_size=1)

    for data in dataList:
        if isDebug:
            data.save_to_img(save_path=f"{PATH_PADDLE_FILE_OUTPUT}table/wired/{count}.jpg")
            data.save_to_json(save_path=f"{PATH_PADDLE_FILE_OUTPUT}table/wired/{count}.json")

        _processTable("wired", data, input, count)

def _extractTableCell(data, input, count):
    labelNameList = data.get("label_names", [])
    scoreList = data.get("scores", [])

    if len(labelNameList) == len(scoreList):
        resultIndex = int(numpy.argmax(scoreList))
        resultLabel = labelNameList[resultIndex]

        if (resultLabel == "wired_table"):
            if isDebug:
                cv2.imwrite(f"{PATH_PADDLE_FILE_OUTPUT}table/wired/{count}_crop.jpg", input)

            _inferenceTableWired(input, count)
        elif (resultLabel == "wireless_table"):
            if isDebug:
                cv2.imwrite(f"{PATH_PADDLE_FILE_OUTPUT}table/wireless/{count}_crop.jpg", input)

            _inferenceTableWireless(input, count)

def _inferenceTableClassification(input, count):
    model = TableClassification(model_dir=pathModelTableClassification, model_name="PP-LCNet_x1_0_table_cls", device=device)
    dataList = model.predict(input=input, batch_size=1)

    for data in dataList:
        if isDebug:
            data.save_to_img(save_path=f"{PATH_PADDLE_FILE_OUTPUT}table/classification/{count}.jpg")
            data.save_to_json(save_path=f"{PATH_PADDLE_FILE_OUTPUT}table/classification/{count}.json")

    _extractTableCell(data, input, count)

def _extractTable(data, input):
    boxList = data.get("boxes", [])

    count = 0

    for box in boxList:
        if str(box.get("label", "")).lower() != "table":
            continue

        coordinateList = box.get("coordinate", [])
        x1, y1, x2, y2 = map(int, coordinateList)

        inputCrop = input[y1:y2, x1:x2, :]

        _inferenceTableClassification(inputCrop, count)

        count += 1

def removeOutputDir():
    if os.path.exists(f"{PATH_PADDLE_FILE_OUTPUT}layout/"):
        shutil.rmtree(f"{PATH_PADDLE_FILE_OUTPUT}layout/")

    if os.path.exists(f"{PATH_PADDLE_FILE_OUTPUT}table/"):
        shutil.rmtree(f"{PATH_PADDLE_FILE_OUTPUT}table/")

    if os.path.exists(f"{PATH_PADDLE_FILE_OUTPUT}export/"):
        shutil.rmtree(f"{PATH_PADDLE_FILE_OUTPUT}export/")

def createOutputDir():
    os.makedirs(f"{PATH_PADDLE_FILE_OUTPUT}layout/", exist_ok=True)
    os.makedirs(f"{PATH_PADDLE_FILE_OUTPUT}table/classification/", exist_ok=True)
    os.makedirs(f"{PATH_PADDLE_FILE_OUTPUT}table/wired/", exist_ok=True)
    os.makedirs(f"{PATH_PADDLE_FILE_OUTPUT}table/wireless/", exist_ok=True)
    os.makedirs(f"{PATH_PADDLE_FILE_OUTPUT}export/", exist_ok=True)

def preprocess(pathImage, sizeLimit=2048):
    imageInput = Image.open(pathImage)
    imageInput = numpy.array(imageInput)

    height, width = imageInput.shape[:2]
    maxSide = max(height, width)

    if maxSide > sizeLimit:
        scale = sizeLimit / maxSide
        newWidth = int(width * scale)
        newHeight = int(height * scale)
        imageInput = cv2.resize(imageInput, (newWidth, newHeight))

    return imageInput

def inferenceLayout(input):
    model = LayoutDetection(model_dir=pathModelLayout, model_name="PP-DocLayout_plus-L", device=device)
    dataList = model.predict(input=input, batch_size=1)

    for data in dataList:
        if isDebug:
            data.save_to_img(save_path=f"{PATH_PADDLE_FILE_OUTPUT}layout/result.jpg")
            data.save_to_json(save_path=f"{PATH_PADDLE_FILE_OUTPUT}layout/result.json")

        _extractTable(data, input)

# Execution
#removeOutputDir()

#createOutputDir()

#imageInput = preprocess("/home/app/file/input/1_jp.jpg")

#inferenceLayout(imageInput)

with open(f"{PATH_PADDLE_FILE_OUTPUT}table/wired/1_result.json", "r", encoding="utf-8") as file:
    data = json.load(file)

_jsonToExcel(data, 14, f"{PATH_PADDLE_FILE_OUTPUT}table/wired/1_result.xlsx")