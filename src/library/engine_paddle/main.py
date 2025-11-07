import os
import logging
import ast
import json
import shutil
from paddleocr import LayoutDetection, TableClassification, TableCellsDetection, TextDetection, TextRecognition

# Source
from image_processor import main as imageProcessor
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
IS_DEBUG = _checkEnvVariable("MS_O_IS_DEBUG")
PATH_FILE = _checkEnvVariable("MS_O_PATH_FILE")

class EnginePaddle:
    def _textOverlapCell(self, textBbox, cellBbox, overlapThreshold=0.5):
        textX, textY, textW, textH = textBbox
        textX1, textY1 = textX, textY
        textX2, textY2 = textX + textW, textY + textH
        
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
            return (overlapArea / textArea) > overlapThreshold
        
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

    def _processTable(self, mode, data, imageOpen, count):
        resultMergeList = []
        pilFont = None

        if IS_DEBUG:
            pilImage, pilImageDraw = imageProcessor.pilImage(imageOpen)

        inferenceTextDataList = self._inferenceText(imageOpen, False)

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
            
            if IS_DEBUG:
                imageCrop = imageOpen[top:bottom, left:right, :]

                imageProcessor.write(f"{PATH_ROOT}{PATH_FILE}output/paddle/{self.uniqueId}/table/{mode}/crop/{count}_{index}.jpg", "", imageCrop)
            
            resultMergeList.append({
                "bbox_list": [left, top, right, bottom],
                "text_list": []
            })
        
            for inferenceTextData in inferenceTextDataList:
                bboxList = inferenceTextData["bbox_list"]
                text = inferenceTextData["text"]

                if IS_DEBUG:
                    cellX1 = min(left, right)
                    cellY1 = min(top, bottom)
                    cellX2 = max(left, right)
                    cellY2 = max(top, bottom)
                    pilImageDraw.rectangle([(cellX1, cellY1), (cellX2, cellY2)], outline=(255, 0, 0), width=1)

                    textX1 = bboxList[0]
                    textY1 = bboxList[1]
                    textX2 = bboxList[0] + bboxList[2]
                    textY2 = bboxList[1] + bboxList[3]
                    pilImageDraw.rectangle([(textX1, textY1), (textX2, textY2)], outline=(0, 0, 255), width=1)

                if self._textOverlapCell(bboxList, [left, top, right, bottom]):
                    pilFont = imageProcessor.pilFont(text, bboxList[2], bboxList[3], self.fontName)

                    if IS_DEBUG:
                        pilImageDraw.text((bboxList[0], bboxList[1]), text, font=pilFont, fill=(0, 0, 0))

                    resultMergeList[index]["text_list"].append(text)

        if IS_DEBUG:
            pilImage.save(f"{PATH_ROOT}{PATH_FILE}output/paddle/{self.uniqueId}/table/{mode}/{count}_result.jpg", format="JPEG")

            with open(f"{PATH_ROOT}{PATH_FILE}output/paddle/{self.uniqueId}/table/{mode}/{count}_result.json", "w", encoding="utf-8") as file:
                json.dump(resultMergeList, file, ensure_ascii=False, indent=2)

        DataToTable(resultMergeList, f"{PATH_ROOT}{PATH_FILE}output/paddle/{self.uniqueId}/table/{mode}/{count}_result.xlsx", pilFont)
        DataToTable(resultMergeList, f"{PATH_ROOT}{PATH_FILE}output/paddle/{self.uniqueId}/table/{mode}/{count}_result.html", pilFont)

    def _inferenceTableWireless(self, imageOpen, count):
        dataList = self.tableCellDetectionWireless.predict(input=imageOpen, batch_size=1)

        for data in dataList:
            if IS_DEBUG:
                data.save_to_json(save_path=f"{PATH_ROOT}{PATH_FILE}output/paddle/{self.uniqueId}/table/wireless/{count}.json")

            self._processTable("wireless", data, imageOpen, count)

    def _inferenceTableWired(self, imageOpen, count):
        dataList = self.tableCellDetectionWired.predict(input=imageOpen, batch_size=1)

        for data in dataList:
            if IS_DEBUG:
                data.save_to_json(save_path=f"{PATH_ROOT}{PATH_FILE}output/paddle/{self.uniqueId}/table/wired/{count}.json")

            self._processTable("wired", data, imageOpen, count)

    def _extractTableCell(self, data, imageOpen, count):
        labelNameList = data.get("label_names", [])
        scoreList = data.get("scores", [])

        if len(labelNameList) == len(scoreList):
            resultIndex = max(range(len(scoreList)), key=lambda a: scoreList[a])
            resultLabel = labelNameList[resultIndex]

            if (resultLabel == "wired_table"):
                if IS_DEBUG:
                    imageProcessor.write(f"{PATH_ROOT}{PATH_FILE}output/paddle/{self.uniqueId}/table/wired/{count}.jpg", "", imageOpen)

                self._inferenceTableWired(imageOpen, count)
            elif (resultLabel == "wireless_table"):
                if IS_DEBUG:
                    imageProcessor.write(f"{PATH_ROOT}{PATH_FILE}output/paddle/{self.uniqueId}/table/wireless/{count}.jpg", "", imageOpen)

                self._inferenceTableWireless(imageOpen, count)

    def _inferenceTableClassification(self, imageOpen, count):
        dataList = self.tableClassification.predict(input=imageOpen, batch_size=1)

        for data in dataList:
            if IS_DEBUG:
                data.save_to_json(save_path=f"{PATH_ROOT}{PATH_FILE}output/paddle/{self.uniqueId}/table/classification/{count}.json")

            self._extractTableCell(data, imageOpen, count)

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

                imageCrop = imageOpen[top:bottom, left:right, :]

                self._inferenceTableClassification(imageCrop, count)

                count += 1

    def _inferenceLayout(self, imageOpen):
        dataList = self.layout.predict(input=imageOpen, batch_size=1)

        for data in dataList:
            if IS_DEBUG:
                data.save_to_json(save_path=f"{PATH_ROOT}{PATH_FILE}output/paddle/{self.uniqueId}/layout/{self.fileNameSplit}.json")

            self._extractTable(data, imageOpen)

    def _inferenceText(self, imageOpen, isWriteOutput=True):
        resultMergeList = []
        cropList = []
        bboxList = []

        if isWriteOutput:
            pilImage, pilImageDraw = imageProcessor.pilImage(imageOpen)

        dataList = self.textDetection.predict(input=imageOpen, batch_size=1)

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

                imageCrop = imageOpen[top:bottom, left:right]
                cropList.append(imageCrop)
                bboxList.append([left, top, width, height])
        
        batchSize = 4

        for index in range(0, len(cropList), batchSize):
            cropBatch = cropList[index:index + batchSize]
            dataSubList = self.textRecognition.predict(input=cropBatch, batch_size=len(cropBatch))

            for bboxSubList, dataSub in zip(bboxList[index:index + batchSize], dataSubList):
                text = dataSub.get("rec_text", "")

                if isWriteOutput:
                    left, top, width, height = bboxSubList
                    pilFont = imageProcessor.pilFont(text, width, height, self.fontName)
                    pilImageDraw.text((left, top), text, font=pilFont, fill=(0, 0, 0))

                resultMergeList.append({
                    "bbox_list": bboxSubList,
                    "text": text
                })

        if isWriteOutput:
            pilImage.save(f"{PATH_ROOT}{PATH_FILE}output/paddle/{self.uniqueId}/export/{self.fileNameSplit}_result.pdf", format="PDF")

            with open(f"{PATH_ROOT}{PATH_FILE}output/paddle/{self.uniqueId}/export/{self.fileNameSplit}_result.json", "w", encoding="utf-8") as file:
                json.dump(resultMergeList, file, ensure_ascii=False, indent=2)
        
        return resultMergeList

    def _createOutputDir(self):
        os.makedirs(f"{PATH_ROOT}{PATH_FILE}output/paddle/{self.uniqueId}/layout/", exist_ok=True)
        os.makedirs(f"{PATH_ROOT}{PATH_FILE}output/paddle/{self.uniqueId}/table/classification/", exist_ok=True)
        os.makedirs(f"{PATH_ROOT}{PATH_FILE}output/paddle/{self.uniqueId}/table/wired/", exist_ok=True)
        os.makedirs(f"{PATH_ROOT}{PATH_FILE}output/paddle/{self.uniqueId}/table/wired/crop/", exist_ok=True)
        os.makedirs(f"{PATH_ROOT}{PATH_FILE}output/paddle/{self.uniqueId}/table/wireless/", exist_ok=True)
        os.makedirs(f"{PATH_ROOT}{PATH_FILE}output/paddle/{self.uniqueId}/table/wireless/crop/", exist_ok=True)
        os.makedirs(f"{PATH_ROOT}{PATH_FILE}output/paddle/{self.uniqueId}/export/", exist_ok=True)

    def _execute(self, _, fileNameValue, uniqueIdValue):
        self.fileName = fileNameValue
        self.fileNameSplit = ".".join(self.fileName.split(".")[:-1])
        self.uniqueId = uniqueIdValue
        
        self._createOutputDir()

        imageOpen, _, _ = imageProcessor.open(f"{PATH_ROOT}{PATH_FILE}input/{self.fileName}")
        
        self._inferenceText(imageOpen)

        self._inferenceLayout(imageOpen)

        print("ok", flush=True)

    def __init__(self):
        self.fileName = ""
        self.uniqueId = ""
        self.fontName = "NotoSansJP-Regular.ttf"
        
        self.device = "gpu" if shutil.which("nvidia-smi") is not None else "cpu"

        self.modelDetectionName = "PP-OCRv5_mobile_det" if self.device == "cpu" else "PP-OCRv5_server_det"
        self.modelRecognitionName = "PP-OCRv5_mobile_rec" if self.device == "cpu" else "PP-OCRv5_server_rec"

        pathModelTextDetection = f"{PATH_ROOT}src/library/engine_paddle/{self.modelDetectionName}/"
        pathModelTextRecognition = f"{PATH_ROOT}src/library/engine_paddle/{self.modelRecognitionName}/"
        pathModelLayout = f"{PATH_ROOT}src/library/engine_paddle/PP-DocLayout_plus-L/"
        pathModelTableClassification = f"{PATH_ROOT}src/library/engine_paddle/PP-LCNet_x1_0_table_cls/"
        pathModelTableWired = f"{PATH_ROOT}src/library/engine_paddle/RT-DETR-L_wired_table_cell_det/"
        pathModelTableWireless = f"{PATH_ROOT}src/library/engine_paddle/RT-DETR-L_wireless_table_cell_det/"

        self.textDetection = TextDetection(model_dir=pathModelTextDetection, model_name=self.modelDetectionName, device=self.device)
        self.textRecognition = TextRecognition(model_dir=pathModelTextRecognition, model_name=self.modelRecognitionName, device=self.device)
        self.layout = LayoutDetection(model_dir=pathModelLayout, model_name="PP-DocLayout_plus-L", device=self.device)
        self.tableClassification = TableClassification(model_dir=pathModelTableClassification, model_name="PP-LCNet_x1_0_table_cls", device=self.device)
        self.tableCellDetectionWired = TableCellsDetection(model_dir=pathModelTableWired, model_name="RT-DETR-L_wired_table_cell_det", device=self.device)
        self.tableCellDetectionWireless = TableCellsDetection(model_dir=pathModelTableWireless, model_name="RT-DETR-L_wireless_table_cell_det", device=self.device)
