import os
import logging
import ast
import json
from paddleocr import LayoutDetection, TableClassification, TableCellsDetection, PaddleOCR

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
    def _filterOverlapBbox(self, bboxList):
        resultList = []

        for bbox in bboxList:
            x1, y1, x2, y2 = [int(bboxCoordinate) for bboxCoordinate in bbox["coordinate"]]
            
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

            resultList.append(bbox)
        
        return resultList

    def _processTable(self, mode, data, input, count):
        if self.isDebug:
            pilImage, pilImageDraw, pilFont = cv2Processor.pilImage(input, self.fontName, 14)
        
        resultMergeList = []

        bboxList = data.get("boxes", [])
        boxFilterList = self._filterOverlapBbox(bboxList)

        for bbox in boxFilterList:
            coordinateList = bbox.get("coordinate", [])
            coordinateList = [int(coordinate) for coordinate in coordinateList]
            
            x1, y1, x2, y2 = coordinateList

            inputCrop = input[y1:y2, x1:x2, :]

            predictDataList = self.ocr.predict(input=inputCrop)
            
            for predictData in predictDataList:
                textList = predictData.get("rec_texts", []) or [""]

                if self.isDebug:
                    lineNumber = len(textList)
                    boxHeight = y2 - y1

                    textBboxList = [pilImageDraw.textbbox((0, 0), text, font=pilFont) for text in textList]
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

                        pilImageDraw.text((textPositionX, textPositionY), text, font=pilFont, fill=(0, 0, 0))

                        currentY += textHeight + extraSpace

                resultMergeList.append({
                    "bbox_list": coordinateList,
                    "text_list": textList
                })

        if self.isDebug:
            pilImage.save(f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/table/{mode}/{count}_result.jpg", format="JPEG")

            with open(f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/table/{mode}/{count}_result.json", "w", encoding="utf-8") as file:
                json.dump(resultMergeList, file, ensure_ascii=False, indent=2)

        DataToTable(resultMergeList, f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/table/{mode}/{count}_result.xlsx")

    def _inferenceTableWireless(self, input, count):
        model = TableCellsDetection(model_dir=self.pathModelTableWireless, model_name="RT-DETR-L_wireless_table_cell_det", device=self.device)
        dataList = model.predict(input=input, batch_size=1)

        for data in dataList:
            if self.isDebug:
                data.save_to_json(save_path=f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/table/wireless/{count}.json")

            self._processTable("wireless", data, input, count)

    def _inferenceTableWired(self, input, count):
        model = TableCellsDetection(model_dir=self.pathModelTableWired, model_name="RT-DETR-L_wired_table_cell_det", device=self.device)
        dataList = model.predict(input=input, batch_size=1)

        for data in dataList:
            if self.isDebug:
                data.save_to_json(save_path=f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/table/wired/{count}.json")

            self._processTable("wired", data, input, count)

    def _extractTableCell(self, data, input, count):
        labelNameList = data.get("label_names", [])
        scoreList = data.get("scores", [])

        if len(labelNameList) == len(scoreList):
            resultIndex = max(range(len(scoreList)), key=lambda a: scoreList[a])
            resultLabel = labelNameList[resultIndex]

            if (resultLabel == "wired_table"):
                if self.isDebug:
                    cv2Processor.write(f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/table/wired/{count}.jpg", "_crop", input)

                self._inferenceTableWired(input, count)
            elif (resultLabel == "wireless_table"):
                if self.isDebug:
                    cv2Processor.write(f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/table/wireless/{count}.jpg", "_crop", input)

                self._inferenceTableWireless(input, count)

    def _inferenceTableClassification(self, input, count):
        model = TableClassification(model_dir=self.pathModelTableClassification, model_name="PP-LCNet_x1_0_table_cls", device=self.device)
        dataList = model.predict(input=input, batch_size=1)

        for data in dataList:
            if self.isDebug:
                data.save_to_json(save_path=f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/table/classification/{count}.json")

            self._extractTableCell(data, input, count)

    def _extractTable(self, data, input):
        bboxList = data.get("boxes", [])
        boxFilterList = self._filterOverlapBbox(bboxList)

        count = 0

        for bbox in boxFilterList:
            label = str(bbox.get("label", "")).lower()

            if label == "table":
                coordinateList = bbox.get("coordinate", [])

                x1, y1, x2, y2 = [int(coordinate) for coordinate in coordinateList]

                inputCrop = input[y1:y2, x1:x2, :]

                self._inferenceTableClassification(inputCrop, count)

                count += 1

    def _inferenceLayout(self, input):
        model = LayoutDetection(model_dir=self.pathModelLayout, model_name="PP-DocLayout_plus-L", device=self.device)
        dataList = model.predict(input=input, batch_size=1)

        for data in dataList:
            if self.isDebug:
                data.save_to_json(save_path=f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/layout/{self.fileNameSplit}.json")

            self._extractTable(data, input)

    def _inferenceText(self, input):
        pilImage, pilImageDraw, pilFont = cv2Processor.pilImage(input, self.fontName, 14)

        resultMergeList = []
        
        predictDataList = self.ocr.predict(input=input)
        
        for predictData in predictDataList:
            bboxGroupList = predictData.get("rec_boxes", [])
            textList = predictData.get("rec_texts", []) or [""]
            
            for index in range(len(bboxGroupList)):
                bboxList = [int(coord) for coord in bboxGroupList[index]]
                text = textList[index]

                x1, y1, _, _ = bboxList
                
                pilImageDraw.text((x1, y1), text, font=pilFont, fill=(0, 0, 0))

                resultMergeList.append({
                    "bbox_list": bboxList,
                    "text": text
                })

        pilImage.save(f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/export/{self.fileNameSplit}_result.pdf", format="PDF")

        with open(f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/export/{self.fileNameSplit}_result.json", "w", encoding="utf-8") as file:
            json.dump(resultMergeList, file, ensure_ascii=False, indent=2)

    def _createOutputDir(self):
        os.makedirs(f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/layout/", exist_ok=True)
        os.makedirs(f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/table/classification/", exist_ok=True)
        os.makedirs(f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/table/wired/", exist_ok=True)
        os.makedirs(f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/table/wireless/", exist_ok=True)
        os.makedirs(f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/export/", exist_ok=True)

    def _execute(self):
        self.ocr = PaddleOCR(
            text_detection_model_dir=self.pathModelTextDetection,
            text_detection_model_name=self.modelDetectionName,
            text_recognition_model_dir=self.pathModelTextRecognition,
            text_recognition_model_name=self.modelRecognitionName,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=self.device
        )

        self._createOutputDir()

        imageOpen, _, _ = cv2Processor.open(f"{PATH_ROOT}{PATH_FILE_INPUT}{self.fileName}")

        self._inferenceText(imageOpen)

        self._inferenceLayout(imageOpen)

    def __init__(self, languageValue, fileNameValue, isCuda, isDebugValue, uniqueIdValue):
        self.language = languageValue
        self.fileName = fileNameValue
        self.device = "gpu" if isCuda else "cpu"
        self.isDebug = isDebugValue
        self.uniqueId = uniqueIdValue

        self.modelDetectionName = "PP-OCRv5_mobile_det" if self.device == "cpu" else "PP-OCRv5_server_det"
        self.modelRecognitionName = "PP-OCRv5_mobile_rec" if self.device == "cpu" else "PP-OCRv5_server_rec"

        self.pathModelLayout = f"{PATH_ROOT}src/library/engine_paddle/PP-DocLayout_plus-L/"
        self.pathModelTableClassification = f"{PATH_ROOT}src/library/engine_paddle/PP-LCNet_x1_0_table_cls/"
        self.pathModelTableWired = f"{PATH_ROOT}src/library/engine_paddle/RT-DETR-L_wired_table_cell_det/"
        self.pathModelTableWireless = f"{PATH_ROOT}src/library/engine_paddle/RT-DETR-L_wireless_table_cell_det/"
        self.pathModelTextDetection = f"{PATH_ROOT}src/library/engine_paddle/{self.modelDetectionName}/"
        self.pathModelTextRecognition = f"{PATH_ROOT}src/library/engine_paddle/{self.modelRecognitionName}/"

        self.fontName = "NotoSansCJK-Regular.ttc"

        self.fileNameSplit = ".".join(self.fileName.split(".")[:-1])

        self._execute()

        print("ok", flush=True)
