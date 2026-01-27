import os
import logging
import ast
import json
import cv2
import numpy
import unicodedata

# Source
from .ppocr_detection import PPocrDetection
from .crnn_recognition import CrnnRecognition
from image_processor import main as imageProcessor

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

class EngineRealtime:
    def _checkMatch(self, value):
        if not self.searchText or value is None:
            return False
        
        text = unicodedata.normalize("NFKC", value).strip()
        searchText = unicodedata.normalize("NFKC", self.searchText).strip()

        if not self.argumentCaseSensitive:
            text = text.casefold()
            searchText = searchText.casefold()

        if not self.argumentSelective:            
            text = text.replace(" ", "")
            searchText = searchText.replace(" ", "")
        
        if self.argumentContain:
            return searchText in text

        return text == searchText

    def _inference(self, imageRgb, detector, recognizer):
        resultMatchList = []

        if imageRgb.ndim == 2:
            imageRgb = cv2.cvtColor(imageRgb, cv2.COLOR_GRAY2RGB)

        imageBgr = cv2.cvtColor(imageRgb, cv2.COLOR_RGB2BGR)

        originalHeight, originalWidth = imageRgb.shape[:2]
        imageResize = cv2.resize(imageBgr, (736, 736))
        scaleWidth = originalWidth / 736.0
        scaleHeight = originalHeight / 736.0

        resultDetector, _ = detector.infer(imageResize)
        textList = [recognizer.infer(imageResize, box.reshape(8)) for box in resultDetector]

        left = cv2.cvtColor(imageBgr, cv2.COLOR_BGR2RGB).copy()
        right = numpy.ones_like(left) * 255

        for boxRaw, text in zip(resultDetector, textList):
            box = numpy.int32([[int(point[0] * scaleWidth), int(point[1] * scaleHeight)] for point in boxRaw])

            xs = box[:, 0]
            boxWidth = xs.max() - xs.min()
            height1 = numpy.linalg.norm(box[1] - box[0])
            hegith2 = numpy.linalg.norm(box[2] - box[3])
            boxHeight = (height1 + hegith2) / 2.0

            (_, textHeight), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 1)
            fontScale = (boxHeight * 0.8) / textHeight if textHeight > 0 else 1.0
            fontThickness = max(1, int(fontScale))
            
            (textWidth, textHeight), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale, fontThickness)
            scaleX = boxWidth / textWidth if textWidth > 0 else 1.0
            scaleY = (boxHeight * 0.8) / textHeight if textHeight > 0 else 1.0
            fontScale = fontScale * min(1.0, scaleX, scaleY)
            fontThickness = max(1, int(numpy.floor(fontScale)))

            if self._checkMatch(text):
                color = (0, 200, 0)

                resultMatchList.append((text, box.copy()))
            else:
                color = (0, 0, 255)

            cv2.polylines(left, [box], isClosed=True, color=color, thickness=1)
            cv2.polylines(right, [box], isClosed=True, color=color, thickness=1)

            x0, y0 = box[0]
            textY = max(0, int(y0 - 5))

            cv2.putText(right, text, (int(x0), textY), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 0), fontThickness, lineType=cv2.LINE_AA)

        return cv2.hconcat([left, right]), resultMatchList

    def _process(self, detector, recognizer):
        resultList = []

        imageRead = cv2.imread(f"{PATH_ROOT}{PATH_FILE}input/{self.fileName}")

        if imageRead is None:
            print(f"Error: Cannot read image: {self.fileName}")

            return None

        imageRgb = cv2.cvtColor(imageRead, cv2.COLOR_BGR2RGB)

        imageResult, matchList = self._inference(imageRgb, detector, recognizer)

        pilImage, _ = imageProcessor.pilImage(cv2.cvtColor(imageResult, cv2.COLOR_RGB2BGR))

        if IS_DEBUG:
            pilImage.save(f"{PATH_ROOT}{PATH_FILE}output/engine_realtime/{self.uniqueId}/{self.fileNameSplit}_result.jpg", format="JPEG")

        if IS_DEBUG and self.searchText is not None and matchList:
            print(f"Match: '{self.searchText}' found {len(matchList)} in {self.fileName}:")

            for a, (text, box) in enumerate(matchList, 1):
                resultList = [(int(x), int(y)) for x, y in box.tolist()]

                print(f"  {a}. text='{text}'  polygon={resultList}")
        
        pilImage.save(f"{PATH_ROOT}{PATH_FILE}output/engine_realtime/{self.uniqueId}/export/{self.fileNameSplit}_result.pdf", format="PDF")

        with open(f"{PATH_ROOT}{PATH_FILE}output/engine_realtime/{self.uniqueId}/export/{self.fileNameSplit}_result.json", "w", encoding="utf-8") as file:
            json.dump(resultList, file, ensure_ascii=False, indent=2)

    def _loadModel(self):
        idBackend = cv2.dnn.DNN_BACKEND_OPENCV
        idTarget = cv2.dnn.DNN_TARGET_CPU

        detector = PPocrDetection(
            modelPath=f"{PATH_ROOT}src/library/engine_realtime/text_detection_en_ppocrv3_2023may.onnx",
            inputSize=[736, 736],
            binaryThreshold=0.3,
            polygonThreshold=0.5,
            maxCandidates=200,
            unclipRatio=2.0,
            backendId=idBackend,
            targetId=idTarget
        )
        recognizer = CrnnRecognition(
            modelPath=f"{PATH_ROOT}src/library/engine_realtime/text_recognition_CRNN_EN_2021sep.onnx",
            backendId=idBackend,
            targetId=idTarget
        )
        return detector, recognizer

    def _createOutputDir(self):
        os.makedirs(f"{PATH_ROOT}{PATH_FILE}output/engine_realtime/{self.uniqueId}/export/", exist_ok=True)

    def _execute(self, languageValue="", fileNameValue="", uniqueIdValue="", searchTextValue=""):
        self.language = languageValue
        self.fileName = fileNameValue
        self.fileNameSplit = ".".join(self.fileName.split(".")[:-1])
        self.uniqueId = uniqueIdValue
        self.searchText = searchTextValue

        self._createOutputDir()
        
        detector, recognizer = self._loadModel()

        self._process(detector, recognizer)

        print("ok", flush=True)

    def __init__(self):
        self.language = ""
        self.fileName = ""
        self.uniqueId = ""
        self.searchText = ""
        self.argumentCaseSensitive = False
        self.argumentSelective = False
        self.argumentContain = False
