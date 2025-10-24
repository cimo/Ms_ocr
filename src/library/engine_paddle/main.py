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
        resultMergeList = []

        if self.isDebug:
            pilImage, pilImageDraw, pilFont = cv2Processor.pilImage(input, self.fontName, 14)
        
        imageInputDebug = input.copy()
        imageInputDebug.fill(255)
        pilImageTest, pilImageDrawTest, pilFontTest = cv2Processor.pilImage(imageInputDebug, self.fontName, 14)

        bboxList = data.get("boxes", [])
        bboxList.sort(key=lambda box: (int(box["coordinate"][1]), int(box["coordinate"][0])))
        #bboxFilterList = self._filterOverlapBbox(bboxList)

        for index, bbox in enumerate(bboxList):
            coordinateList = bbox.get("coordinate", [])
            coordinateList = [int(coordinate) for coordinate in coordinateList]
            
            x1, y1, x2, y2 = coordinateList
            
            height, width = input.shape[:2]
            x1 = max(0, min(x1, width))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height))
            y2 = max(0, min(y2, height))

            if x2 <= x1 or y2 <= y1:
                print("cimo")

            inputCrop = input[y1:y2, x1:x2, :]

            cv2Processor.write(f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/table/{mode}/crop/{count}_{index}.jpg", "", inputCrop)

            resultMergeList.append({
                "bbox_list": coordinateList,
                "text_list": []
            })

            inferenceTextDataList, pilImageDrawTest = self._inferenceText(inputCrop, False, x1, y1, pilImageDrawTest, pilFontTest)
            inferenceTextDataList.sort(key=lambda item: (item["bbox_list"][1], item["bbox_list"][0]))

            for inferenceTextData in inferenceTextDataList:
                bboxSubList = inferenceTextData.get("bbox_list", [])
                text = inferenceTextData.get("text", "")

                if self.isDebug:
                    fontSize = max(10, int(bboxSubList[3] * 0.8))
                    pilFont = ImageFont.truetype(self.fontName, fontSize)

                    pilImageDraw.text((bboxSubList[0], bboxSubList[1]), text, font=pilFont, fill=(0, 0, 0))

                    x, y, _, _ = bboxSubList

                    if x1 <= x <= x2 and y1 <= y <= y2:
                        resultMergeList[index]["text_list"].append(text)

        if self.isDebug:
            pilImage.save(f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/table/{mode}/{count}_result.jpg", format="JPEG")

            with open(f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/table/{mode}/{count}_result.json", "w", encoding="utf-8") as file:
                json.dump(resultMergeList, file, ensure_ascii=False, indent=2)
        
        pilImageTest.save(f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/table/{mode}/{count}_result_test.jpg", format="JPEG")

        #DataToTable(resultMergeList, f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/table/{mode}/{count}_result.xlsx")

    def _inferenceTableWireless(self, input, count):
        dataList = self.tableCellDetectionWireless.predict(input=input, batch_size=1)

        for data in dataList:
            if self.isDebug:
                data.save_to_json(save_path=f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/table/wireless/{count}.json")

            self._processTable("wireless", data, input, count)

    def _inferenceTableWired(self, input, count):
        dataList = self.tableCellDetectionWired.predict(input=input, batch_size=1)

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
                    cv2Processor.write(f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/table/wired/crop/{count}.jpg", "", input)

                self._inferenceTableWired(input, count)
            elif (resultLabel == "wireless_table"):
                if self.isDebug:
                    cv2Processor.write(f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/table/wireless/crop/{count}.jpg", "", input)

                self._inferenceTableWireless(input, count)

    def _inferenceTableClassification(self, input, count):
        dataList = self.tableClassificationInit.predict(input=input, batch_size=1)

        for data in dataList:
            if self.isDebug:
                data.save_to_json(save_path=f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/table/classification/{count}.json")

            self._extractTableCell(data, input, count)

    def _extractTable(self, data, input):
        bboxList = data.get("boxes", [])
        boxFilterList = self._filterOverlapBbox(bboxList)

        count = 0

        for bbox in boxFilterList:
            label = bbox.get("label", "")
            label = str(label).lower()

            if label == "table":
                coordinateList = bbox.get("coordinate", [])

                x1, y1, x2, y2 = [int(coordinate) for coordinate in coordinateList]

                inputCrop = input[y1:y2, x1:x2, :]

                self._inferenceTableClassification(inputCrop, count)

                count += 1

    def _inferenceLayout(self, input):
        dataList = self.layoutInit.predict(input=input, batch_size=1)

        for data in dataList:
            if self.isDebug:
                data.save_to_json(save_path=f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/layout/{self.fileNameSplit}.json")

            self._extractTable(data, input)

    def _inferenceText(self, input, isWriteFileOutput=True, offsetX=0, offsetY=0, pilImageDrawTest=None, pilFontTest=None):
        resultMergeList = []

        if isWriteFileOutput:
            pilImage, pilImageDraw, pilFont = cv2Processor.pilImage(input, self.fontName, 14)
        
        _, _, ratioWidth, ratioHeight, input, _ = cv2Processor.resize(input, 1280, "w")

        dataList = self.textDetectionInit.predict(input=input, batch_size=1)

        for data in dataList:
            if self.isDebug and isWriteFileOutput:
                data.save_to_json(save_path=f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/export/detection.json")

            dtPolyList = data.get("dt_polys", [])
            
            for dtPoly in dtPolyList:
                a = [point[0] for point in dtPoly]
                b = [point[1] for point in dtPoly]

                left = int(min(a))
                top = int(min(b))
                right = int(max(a))
                bottom = int(max(b))
                width = right - left
                height = bottom - top

                inputCrop = input[top:bottom, left:right]

                dataSubList = self.textRecognitionInit.predict(input=inputCrop, batch_size=1)

                for dataSub in dataSubList:
                    if self.isDebug and isWriteFileOutput:
                        dataSub.save_to_json(save_path=f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/export/recognition.json")

                    #if self.isDebug:
                    #    cv2Processor.write(f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/export/{offsetX}_{offsetY}_recognition.jpg", "", inputCrop)
                    
                    text = dataSub.get("rec_text", "")

                    scaleX = 1 / ratioWidth
                    scaleY = 1 / ratioHeight

                    originalLeft = int((offsetX + left) * scaleX)
                    originalTop = int((offsetY + top) * scaleY)
                    originalWidth = int(width * scaleX)
                    originalHeight = int(height * scaleY)

                    if isWriteFileOutput:
                        pilImageDraw.text((originalLeft, originalTop), text, font=pilFont, fill=(0, 0, 0))
                    
                    pilImageDrawTest.rectangle([originalLeft, originalTop, originalLeft + originalWidth, originalTop + originalHeight], outline="red", width=1)
                    pilImageDrawTest.text((originalLeft, originalTop), text, font=pilFontTest, fill=(0, 0, 0))

                    resultMergeList.append({
                        "bbox_list": [originalLeft, originalTop, originalWidth, originalHeight],
                        "text": text
                    })

        if isWriteFileOutput:
            pilImage.save(f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/export/{self.fileNameSplit}_result.pdf", format="PDF")

            with open(f"{PATH_ROOT}{PATH_FILE_OUTPUT}paddle/{self.uniqueId}/export/{self.fileNameSplit}_result.json", "w", encoding="utf-8") as file:
                json.dump(resultMergeList, file, ensure_ascii=False, indent=2)
        
        return resultMergeList, pilImageDrawTest

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

        #self._inferenceText(imageOpen)

        self._inferenceLayout(imageOpen)

    def __init__(self, languageValue, fileNameValue, isCuda, isDebugValue, uniqueIdValue):
        self.language = languageValue
        self.fileName = fileNameValue
        self.device = "gpu" if isCuda else "cpu"
        self.isDebug = isDebugValue
        self.uniqueId = uniqueIdValue

        self.modelDetectionName = "PP-OCRv5_mobile_det" if self.device == "cpu" else "PP-OCRv5_server_det"
        self.modelRecognitionName = "PP-OCRv5_mobile_rec" if self.device == "cpu" else "PP-OCRv5_server_rec"

        self.pathModelTextDetection = f"{PATH_ROOT}src/library/engine_paddle/{self.modelDetectionName}/"
        self.pathModelTextRecognition = f"{PATH_ROOT}src/library/engine_paddle/{self.modelRecognitionName}/"
        self.pathModelLayout = f"{PATH_ROOT}src/library/engine_paddle/PP-DocLayout_plus-L/"
        self.pathModelTableClassification = f"{PATH_ROOT}src/library/engine_paddle/PP-LCNet_x1_0_table_cls/"
        self.pathModelTableWired = f"{PATH_ROOT}src/library/engine_paddle/RT-DETR-L_wired_table_cell_det/"
        self.pathModelTableWireless = f"{PATH_ROOT}src/library/engine_paddle/RT-DETR-L_wireless_table_cell_det/"

        self.fontName = "NotoSansCJK-Regular.ttc"

        self.fileNameSplit = ".".join(self.fileName.split(".")[:-1])

        self.textDetectionInit = TextDetection(model_dir=self.pathModelTextDetection, model_name=self.modelDetectionName, device=self.device)
        self.textRecognitionInit = TextRecognition(model_dir=self.pathModelTextRecognition, model_name=self.modelRecognitionName, device=self.device)
        self.layoutInit = LayoutDetection(model_dir=self.pathModelLayout, model_name="PP-DocLayout_plus-L", device=self.device)
        self.tableClassificationInit = TableClassification(model_dir=self.pathModelTableClassification, model_name="PP-LCNet_x1_0_table_cls", device=self.device)
        self.tableCellDetectionWired = TableCellsDetection(model_dir=self.pathModelTableWired, model_name="RT-DETR-L_wired_table_cell_det", device=self.device)
        self.tableCellDetectionWireless = TableCellsDetection(model_dir=self.pathModelTableWireless, model_name="RT-DETR-L_wireless_table_cell_det", device=self.device)

        self._execute()

        print("ok", flush=True)
