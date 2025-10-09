import sys
import os
import shutil
import json
import cv2
from paddleocr import LayoutDetection, TableClassification, TableCellsDetection, PaddleOCR
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
pathModelTextDetection = f"{PATH_PADDLE_LIBRARY}PP-OCRv5_mobile_det/"
pathModelTextRecognition = f"{PATH_PADDLE_LIBRARY}PP-OCRv5_mobile_rec/"

isDebug = True
device = "cpu"

ocr = PaddleOCR(
    text_detection_model_dir=pathModelTextDetection,
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_dir=pathModelTextRecognition,
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device=device
)

def _filterOverlapBbox(bboxList):
    resultList = []

    for bbox in bboxList:
        x1, y1, x2, y2 = [float(bboxCoordinate) for bboxCoordinate in bbox["coordinate"]]
        
        isIgnore = False
        
        for result in resultList:
            xf1, yf1, xf2, yf2 = [float(resultCoordinate) for resultCoordinate in result["coordinate"]]

            overlapX = max(0, min(x2, xf2) - max(x1, xf1))
            overlapY = max(0, min(y2, yf2) - max(y1, yf1))

            minWidth = min(x2 - x1, xf2 - xf1)
            minHeight = min(y2 - y1, yf2 - yf1)

            if overlapX / minWidth > 0.9 and overlapY / minHeight > 0.9:
                isIgnore = True

                break

        if isIgnore:
            continue

        resultList.append(bbox)
    
    return resultList

def _processTable(mode, data, input, count):
    if isDebug:
        inputHeight, inputWidth = input.shape[:2]
        resultImage = Image.new("RGB", (inputWidth, inputHeight), (255, 255, 255))
        imageDraw = ImageDraw.Draw(resultImage)
        font = ImageFont.truetype(pathFont, 14)
    
    resultMergeList = []

    bboxList = data.get("boxes", [])
    boxFilterList = _filterOverlapBbox(bboxList)

    for bbox in boxFilterList:
        coordinateList = bbox.get("coordinate", [])
        coordinateList = [float(coordinate) for coordinate in coordinateList]
        
        x1, y1, x2, y2 = coordinateList

        inputCrop = input[y1:y2, x1:x2, :]

        predictDataList = ocr.predict(input=inputCrop)
        
        for predictData in predictDataList:
            textList = predictData.get("rec_texts", []) or [""]

            if isDebug:
                lineNumber = len(textList)
                boxHeight = y2 - y1

                textBboxList = [imageDraw.textbbox((0, 0), text, font=font) for text in textList]
                textWidthList = [textBox[2] - textBox[0] for textBox in textBboxList]
                textHeightList = [textBox[3] - textBox[1] for textBox in textBboxList]

                totalTextHeight = sum(textHeightList)

                if lineNumber > 1:
                    extraSpace = (boxHeight - totalTextHeight) / (lineNumber + 1)
                else:
                    extraSpace = (boxHeight - totalTextHeight) / 2

                currentY = y1 + extraSpace

                for index, text in enumerate(textList):
                    textWidth = textWidthList[index]
                    textHeight = textHeightList[index]
                    textPositionX = x1 + (x2 - x1 - textWidth) // 2
                    textPositionY = int(currentY)

                    imageDraw.text((textPositionX, textPositionY), text, font=font, fill=(0, 0, 0))

                    currentY += textHeight + extraSpace

            resultMergeList.append({
                "bbox_list": coordinateList,
                "text_list": textList
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
    bboxList = data.get("boxes", [])
    boxFilterList = _filterOverlapBbox(bboxList)

    count = 0

    for bbox in boxFilterList:
        label = str(bbox.get("label", "")).lower()

        if label == "table":
            coordinateList = bbox.get("coordinate", [])

            x1, y1, x2, y2 = [float(coordinate) for coordinate in coordinateList]

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

def inferenceText(input):
    if isDebug:
        inputHeight, inputWidth = input.shape[:2]
        resultImage = Image.new("RGB", (inputWidth, inputHeight), (255, 255, 255))
        imageDraw = ImageDraw.Draw(resultImage)
        font = ImageFont.truetype("NotoSansCJK-Regular.ttc", 14)
    
    predictDataList = ocr.predict(input=input)
    
    for predictData in predictDataList:
        bboxList = predictData.get("rec_boxes", [])
        textList = predictData.get("rec_texts", []) or [""]

        if isDebug:
            assert len(bboxList) == len(textList), f"Mismatch tra box e testi: {len(bboxList)} vs {len(textList)}"

            for index in range(len(bboxList)):
                print(f"[DEBUG] Box {index}: {bboxList[index]}")
                print(f"[DEBUG] Testo {index}: {textList[index]}")

                assert len(bboxList[index]) == 4, f"Formato bbox non valido: {bboxList[index]}"
                assert bboxList[index][2] > bboxList[index][0] and bboxList[index][3] > bboxList[index][1], f"Box malformato: {bboxList[index]}"

                x1, y1, _, _ = [int(bbox) for bbox in bboxList[index]]

                imageDraw.text((x1, y1), textList[index], font=font, fill=(0, 0, 0))

        predictData.save_to_json(save_path=f"{PATH_PADDLE_FILE_OUTPUT}export/result_recognition.json")
    
    if isDebug:
        resultImage.save(f"{PATH_PADDLE_FILE_OUTPUT}export/result_recognition.jpg", format="JPEG")

def inferenceLayout(input):
    model = LayoutDetection(model_dir=pathModelLayout, model_name="PP-DocLayout_plus-L", device=device)
    dataList = model.predict(input=input, batch_size=1)

    for data in dataList:
        if isDebug:
            data.save_to_img(save_path=f"{PATH_PADDLE_FILE_OUTPUT}layout/result.jpg")
            data.save_to_json(save_path=f"{PATH_PADDLE_FILE_OUTPUT}layout/result.json")

        _extractTable(data, input)

def Main():
    #removeOutputDir()

    #createOutputDir()

    image = preprocessor.open("/home/app/file/input/1_jp.jpg")

    #inferenceLayout(image)
    
    inferenceText(image)

Main()
