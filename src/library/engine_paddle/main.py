import os
import logging
import ast
import json
from PIL import ImageFont
from paddleocr import LayoutDetection, TableClassification, TableCellsDetection, TextDetection, TextRecognition

# Source
from cv2_processor import main as cv2Processor
from data_to_table.main import DataToTable

def _checkEnvVariable(varKey):
    if os.environ.get(varKey) is None:
        logging.exception("Environment variable %s do not exist", varKey)
    else:
        if os.environ.get(varKey).lower() == "true":
            return True
        if os.environ.get(varKey).lower() == "false":
            return False
        if os.environ.get(varKey).isnumeric():
            return int(os.environ.get(varKey))
        if os.environ.get(varKey).startswith("[") and os.environ.get(varKey).endswith("]"):
            return ast.literal_eval(os.environ.get(varKey))

    return os.environ.get(varKey)

ENV_NAME = _checkEnvVariable("ENV_NAME")
PATH_ROOT = _checkEnvVariable("PATH_ROOT")
PATH_FILE_INPUT = _checkEnvVariable("MS_O_PATH_FILE_INPUT")
PATH_FILE_OUTPUT = _checkEnvVariable("MS_O_PATH_FILE_OUTPUT")

class EnginePaddle:
    def _textOverlapCell(self, textBbox, cellBbox, overlapThreshold=0.7):
        textX, textY, textW, textH = textBbox
        textX1 = textX
        textY1 = textY
        textX2 = textX + textW
        textY2 = textY + textH

        cellX1, cellY1, cellX2, cellY2 = cellBbox

        overlapX1 = max(textX1, cellX1)
        overlapY1 = max(textY1, cellY1)
        overlapX2 = min(textX2, cellX2)
        overlapY2 = min(textY2, cellY2)

        overlapWidth = max(0, overlapX2 - overlapX1)
        overlapHeight = max(0, overlapY2 - overlapY1)
        overlapArea = overlapWidth * overlapHeight

        textArea = textW * textH

        if textArea > 0:
            return (overlapArea / textArea) >= overlapThreshold

        return False

    def _filterOverlapBox(self, boxList):
        resultList = []

        for box in boxList:
            x1, y1, x2, y2 = [int(boxCoordinate) for boxCoordinate in box["coordinate"]]
            
            isIgnore = False
            
            for result in resultList:
                xf1, yf1, xf2, yf2 = [int(resultCoordinate) for resultCoordinate in result["coordinate"]]

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

    def _processTable(self, mode, data, imageOpenCrop, count):
        resultMergeList = []

        if self.isDebug:
            pilImage, pilImageDraw, pilFont = cv2Processor.pilImage(imageOpenCrop, self.fontName, 14)

        inferenceTextDataList = self._inferenceText(imageOpenCrop, False)

        boxList = data.get("boxes", [])
        boxFilterList = self._filterOverlapBox(boxList)

        for index, box in enumerate(boxFilterList):
            coordinateList = box.get("coordinate", [])
            x = [int(coordinateList[0]), int(coordinateList[2])]
            y = [int(coordinateList[1]), int(coordinateList[3])]

            left = int(min(x))
            top = int(min(y))
            right = int(max(x))
            bottom = int(max(y))
            #width = right - left
            #height = bottom - top
            
            if self.isDebug:
                crop = imageOpenCrop[top:bottom, left:right, :]

                cv2Processor.write(f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/table/{mode}/crop/{count}_{index}.jpg", "", crop)
            
            resultMergeList.append({
                "bbox_list": [left, top, right, bottom],
                "text_list": []
            })
        
            for inferenceTextData in inferenceTextDataList:
                bboxList = inferenceTextData["bbox_list"]
                text = inferenceTextData["text"]

                if self.isDebug:
                    cx1 = min(left, right)
                    cy1 = min(top, bottom)
                    cx2 = max(left, right)
                    cy2 = max(top, bottom)
                    pilImageDraw.rectangle([(cx1, cy1), (cx2, cy2)], outline=(255, 0, 0), width=1)

                    tx1 = bboxList[0]
                    ty1 = bboxList[1]
                    tx2 = bboxList[0] + bboxList[2]
                    ty2 = bboxList[1] + bboxList[3]
                    pilImageDraw.rectangle([(tx1, ty1), (tx2, ty2)], outline=(0, 0, 255), width=1)

                if self._textOverlapCell(bboxList, [left, top, right, bottom]):
                    if self.isDebug:
                        #fontSize = max(20, int(min(width, height) * 0.2)) 
                        #pilFont = ImageFont.truetype(self.fontName, fontSize)
                        pilImageDraw.text((bboxList[0], bboxList[1]), text, font=pilFont, fill=(0, 0, 0))

                    resultMergeList[index]["text_list"].append(text)

        if self.isDebug:
            pilImage.save(f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/table/{mode}/{count}_result.jpg", format="JPEG")

            with open(f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/table/{mode}/{count}_result.json", "w", encoding="utf-8") as file:
                json.dump(resultMergeList, file, ensure_ascii=False, indent=2)

        DataToTable(resultMergeList, f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/table/{mode}/{count}_result.xlsx")

    def _inferenceTableWireless(self, imageOpenCrop, count):
        dataList = self.tableCellDetectionWireless.predict(input=imageOpenCrop, batch_size=1)

        for data in dataList:
            if self.isDebug:
                data.save_to_json(save_path=f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/table/wireless/{count}.json")

            self._processTable("wireless", data, imageOpenCrop, count)

    def _inferenceTableWired(self, imageOpenCrop, count):
        dataList = self.tableCellDetectionWired.predict(input=imageOpenCrop, batch_size=1)

        for data in dataList:
            if self.isDebug:
                data.save_to_json(save_path=f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/table/wired/{count}.json")

            self._processTable("wired", data, imageOpenCrop, count)

    def _extractTableCell(self, data, imageOpenCrop, count):
        labelNameList = data.get("label_names", [])
        scoreList = data.get("scores", [])

        if len(labelNameList) == len(scoreList):
            resultIndex = max(range(len(scoreList)), key=lambda a: scoreList[a])
            resultLabel = labelNameList[resultIndex]

            if (resultLabel == "wired_table"):
                if self.isDebug:
                    cv2Processor.write(f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/table/wired/{count}.jpg", "", imageOpenCrop)

                self._inferenceTableWired(imageOpenCrop, count)
            elif (resultLabel == "wireless_table"):
                if self.isDebug:
                    cv2Processor.write(f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/table/wireless/{count}.jpg", "", imageOpenCrop)

                self._inferenceTableWireless(imageOpenCrop, count)

    def _inferenceTableClassification(self, imageOpenCrop, count):
        dataList = self.tableClassificationInit.predict(input=imageOpenCrop, batch_size=1)

        for data in dataList:
            if self.isDebug:
                data.save_to_json(save_path=f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/table/classification/{count}.json")

            self._extractTableCell(data, imageOpenCrop, count)

    def _extractTable(self, data, imageOpen):
        boxList = data.get("boxes", [])
        boxFilterList = self._filterOverlapBox(boxList)

        count = 0

        for box in boxFilterList:
            label = box.get("label", "")
            label = str(label).lower()

            if label == "table":
                coordinateList = box.get("coordinate", [])    
                x = [int(coordinateList[0]), int(coordinateList[2])]
                y = [int(coordinateList[1]), int(coordinateList[3])]

                left = int(min(x))
                top = int(min(y))
                right = int(max(x))
                bottom = int(max(y))

                imageOpenCrop = imageOpen[top:bottom, left:right, :]

                resize = cv2Processor.resizeToMultipleRatio(imageOpenCrop)

                self._inferenceTableClassification(resize["result"], count)

                count += 1

    def _inferenceLayout(self, imageOpen):
        dataList = self.layoutInit.predict(input=imageOpen, batch_size=1)

        for data in dataList:
            if self.isDebug:
                data.save_to_json(save_path=f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/layout/{self.fileNameSplit}.json")

            self._extractTable(data, imageOpen)

    def _inferenceText(self, image, isWriteOutput=True):
        resultMergeList = []

        if isWriteOutput:
            image = cv2Processor.resizeToMultipleRatio(image)["result"]

            pilImage, pilImageDraw, _ = cv2Processor.pilImage(image, self.fontName, 14)

        dataList = self.textDetectionInit.predict(input=image, batch_size=1)

        for data in dataList:
            dtPolyList = data.get("dt_polys", [])
            
            for dtPoly in dtPolyList:
                x = [point[0] for point in dtPoly]
                y = [point[1] for point in dtPoly]

                left = int(min(x))
                top = int(min(y))
                right = int(max(x))
                bottom = int(max(y))
                width = right - left
                height = bottom - top

                imageCrop = image[top:bottom, left:right, :]

                dataSubList = self.textRecognitionInit.predict(input=imageCrop, batch_size=1)

                for dataSub in dataSubList:
                    text = dataSub.get("rec_text", "")

                    if isWriteOutput:
                        fontScaleMax = 1 if len(text) <= 2 else 0.8
                        fontSize = min(height, int(height * fontScaleMax))

                        while fontSize > 1:
                            pilFont = ImageFont.truetype(self.fontName, fontSize)
                            bboxText = pilFont.getbbox(text)

                            textWidth = bboxText[2] - bboxText[0]
                            textHeight = bboxText[3] - bboxText[1]

                            if textWidth <= width and textHeight <= height:
                                break

                            fontSize -= 1

                        pilImageDraw.text((left, top), text, font=pilFont, fill=(0, 0, 0))

                    resultMergeList.append({
                        "bbox_list": [left, top, width, height],
                        "text": text
                    })

        if isWriteOutput:
            pilImage.save(f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/export/{self.fileNameSplit}_result.jpg", format="JPEG")

        if self.isDebug and isWriteOutput:
            with open(f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/export/{self.fileNameSplit}_result.json", "w", encoding="utf-8") as file:
                json.dump(resultMergeList, file, ensure_ascii=False, indent=2)
        
        return resultMergeList

    def _createOutputDir(self):
        os.makedirs(f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/layout/", exist_ok=True)
        os.makedirs(f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/table/classification/", exist_ok=True)
        os.makedirs(f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/table/wired/", exist_ok=True)
        os.makedirs(f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/table/wired/crop/", exist_ok=True)
        os.makedirs(f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/table/wireless/", exist_ok=True)
        os.makedirs(f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/table/wireless/crop/", exist_ok=True)
        os.makedirs(f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/export/", exist_ok=True)

    def _execute(self):
        self._createOutputDir()

        imageOpen, _, _ = cv2Processor.open(f"{PATH_ROOT}{PATH_FILE_INPUT}{self.fileName}")

        self._inferenceText(imageOpen)

        #self._inferenceLayout(imageOpen)

    def __init__(self, languageValue, fileNameValue, isCuda, isDebugValue, uniqueIdValue):
        self.language = languageValue
        self.fileName = fileNameValue
        self.device = "gpu" if isCuda else "cpu"
        self.isDebug = isDebugValue
        self.uniqueId = uniqueIdValue

        self.modelRecognitionName = "PP-OCRv5_mobile_rec" if self.device == "cpu" else "PP-OCRv5_server_rec"

        self.pathModelTextDetection = f"{PATH_ROOT}src/library/engine_paddle/PP-OCRv5_server_det/"
        self.pathModelTextRecognition = f"{PATH_ROOT}src/library/engine_paddle/{self.modelRecognitionName}/"
        self.pathModelLayout = f"{PATH_ROOT}src/library/engine_paddle/PP-DocLayout_plus-L/"
        self.pathModelTableClassification = f"{PATH_ROOT}src/library/engine_paddle/PP-LCNet_x1_0_table_cls/"
        self.pathModelTableWired = f"{PATH_ROOT}src/library/engine_paddle/RT-DETR-L_wired_table_cell_det/"
        self.pathModelTableWireless = f"{PATH_ROOT}src/library/engine_paddle/RT-DETR-L_wireless_table_cell_det/"

        self.fontName = "NotoSansCJK-Regular.ttc"

        self.fileNameSplit = ".".join(self.fileName.split(".")[:-1])

        self.textDetectionInit = TextDetection(model_dir=self.pathModelTextDetection, model_name="PP-OCRv5_server_det", device=self.device)
        self.textRecognitionInit = TextRecognition(model_dir=self.pathModelTextRecognition, model_name=self.modelRecognitionName, device=self.device)
        self.layoutInit = LayoutDetection(model_dir=self.pathModelLayout, model_name="PP-DocLayout_plus-L", device=self.device)
        self.tableClassificationInit = TableClassification(model_dir=self.pathModelTableClassification, model_name="PP-LCNet_x1_0_table_cls", device=self.device)
        self.tableCellDetectionWired = TableCellsDetection(model_dir=self.pathModelTableWired, model_name="RT-DETR-L_wired_table_cell_det", device=self.device)
        self.tableCellDetectionWireless = TableCellsDetection(model_dir=self.pathModelTableWireless, model_name="RT-DETR-L_wireless_table_cell_det", device=self.device)

        self._execute()

        print("ok", flush=True)
