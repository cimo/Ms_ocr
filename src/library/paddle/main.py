import sys
import os
import shutil
import json
import cv2
from paddleocr import LayoutDetection, TableClassification, TableCellsDetection, TextDetection, TextRecognition, PaddleOCR
from PIL import Image, ImageDraw, ImageFont

# Source
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from json_to_excel.main import JsonToExcel
from preprocessor import main as preprocessor

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

def _filterOverlapBox(boxList):
    resultList = []

    for box in boxList:
        x1, y1, x2, y2 = map(float, box["coordinate"])
        
        isIgnore = False
        
        for result in resultList:
            xf1, yf1, xf2, yf2 = map(float, result["coordinate"])

            overlapX = max(0, min(x2, xf2) - max(x1, xf1))
            overlapY = max(0, min(y2, yf2) - max(y1, yf1))

            minWidth = min(x2 - x1, xf2 - xf1)
            minHeight = min(y2 - y1, yf2 - yf1)

            if overlapX / minWidth > 0.9 and overlapY / minHeight > 0.9:
                isIgnore = True

                break

        if isIgnore:
            continue

        resultList.append(box)
    
    return resultList

def _inferenceTextRecognition(input):
    model = TextRecognition(model_dir=pathModelTextRecognition, model_name="PP-OCRv5_server_rec")
    datList = model.predict(input=input, batch_size=1)

    for data in datList:
        if isDebug:
            data.save_to_img(save_path=f"{PATH_PADDLE_FILE_OUTPUT}export/result_recognition.jpg")
            data.save_to_json(save_path=f"{PATH_PADDLE_FILE_OUTPUT}export/result_recognition.json")

def _inferenceTextDetection(input):
    model = TextDetection(model_dir=pathModelTextDetection, model_name="PP-OCRv5_server_det")
    datList = model.predict(input=input, batch_size=1)

    for data in datList:
        if isDebug:
            data.save_to_img(save_path=f"{PATH_PADDLE_FILE_OUTPUT}export/result_detection.jpg")
            data.save_to_json(save_path=f"{PATH_PADDLE_FILE_OUTPUT}export/result_detection.json")

        #_inferenceTextRecognition(input)

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

        inputCrop = input[y1:y2, x1:x2, :]
        #_, _, _, inputCropResize, _ = preprocessor.resize(inputCrop, 960)

        resultList = ocr.predict(input=inputCrop)
        
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

                for index, text in enumerate(textList):
                    textWidth = textWidthList[index]
                    textHeight = textHeightList[index]
                    textPositionX = x1 + (x2 - x1 - textWidth) // 2
                    textPositionY = int(currentY)

                    imageDraw.text((textPositionX, textPositionY), text, font=font, fill=(0, 0, 0))

                    currentY += textHeight + extraSpace

            resultMergeList.append({
                "bbox": coordinateList,
                "text": textList
            })

    if isDebug:
        resultImage.save(f"{PATH_PADDLE_FILE_OUTPUT}table/{mode}/{count}_result.jpg", format="JPEG")

        with open(f"{PATH_PADDLE_FILE_OUTPUT}table/{mode}/{count}_result.json", "w", encoding="utf-8") as file:
            json.dump(resultMergeList, file, ensure_ascii=False, indent=2)

    JsonToExcel(isDebug, resultMergeList, f"{PATH_PADDLE_FILE_OUTPUT}table/{mode}/{count}.xlsx")

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
        resultIndex = max(range(len(scoreList)), key=lambda a: scoreList[a])
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
    boxFilterList = _filterOverlapBox(boxList)

    count = 0

    for box in boxFilterList:
        if str(box.get("label", "")).lower() != "table":
            continue

        coordinateList = box.get("coordinate", [])
        x1, y1, x2, y2 = map(int, coordinateList)

        inputCrop = input[y1:y2, x1:x2, :]
        #_, _, _, inputCropResize, _ = preprocessor.resize(inputCrop, 960)

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

def inferenceLayout(input):
    model = LayoutDetection(model_dir=pathModelLayout, model_name="PP-DocLayout_plus-L", device=device)
    dataList = model.predict(input=input, batch_size=1)

    for data in dataList:
        if isDebug:
            data.save_to_img(save_path=f"{PATH_PADDLE_FILE_OUTPUT}layout/result.jpg")
            data.save_to_json(save_path=f"{PATH_PADDLE_FILE_OUTPUT}layout/result.json")

        _extractTable(data, input)

def Main():
    # Execution
    removeOutputDir()

    createOutputDir()

    image = preprocessor.open("/home/app/file/input/1_jp.jpg")
    #_, _, _, imageResize, _ = preprocessor.resize(image, 2048)

    inferenceLayout(image)
    '''
    path = "wired/1_result"

    with open(f"{PATH_PADDLE_FILE_OUTPUT}table/{path}.json", "r", encoding="utf-8") as file:
        dataList = json.load(file)

    JsonToExcel(isDebug, dataList, f"{PATH_PADDLE_FILE_OUTPUT}table/{path}.xlsx")
    '''
Main()
