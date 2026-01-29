import os
import logging
import ast
import json
import cv2
import numpy
import unicodedata

# Source
from .detector_ppocr import DetectorPPocr
from .recognizer_crnn import RecognizerCrnn
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

    def _inference(self, detector, recognizer):
        resultList = []
        
        imageOpen, _, _ = imageProcessor.open(f"{PATH_ROOT}{PATH_FILE}input/{self.fileName}")

        pilImage, pilImageDraw = imageProcessor.pilImage(imageOpen)

        imageBgr = cv2.cvtColor(imageOpen, cv2.COLOR_RGB2BGR)

        originalHeight, originalWidth = imageOpen.shape[:2]
        imageResize = cv2.resize(imageBgr, (detector.inputWidth, detector.inputHeight))
        scaleWidth = originalWidth / float(detector.inputWidth)
        scaleHeight = originalHeight / float(detector.inputHeight)

        resultDetector, _ = detector.infer(imageResize)
        textList = [recognizer.infer(imageResize, box.reshape(8)) for box in resultDetector]

        dataList = []

        for boxRaw, text in zip(resultDetector, textList):
            box = numpy.int32([[int(point[0] * scaleWidth), int(point[1] * scaleHeight)] for point in boxRaw])

            dataList.append((text, box.copy()))

            x, y, w, h = imageProcessor.bbox(box)

            if self._checkMatch(text):
                color = (0, 200, 0)
            else:
                color = (255, 0, 0)

            imageProcessor.rectangle(imageOpen, x, y, w, h, color, 1)

            pilFont = imageProcessor.pilFont(text, w, h, self.fontName)
            pilImageDraw.text((x, y), text, font=pilFont, fill=(0, 0, 0))

        if IS_DEBUG:
            imageProcessor.write(f"{PATH_ROOT}{PATH_FILE}output/engine_realtime/{self.uniqueId}/layout/{self.fileNameSplit}_result.jpg", "", imageOpen)

        for id, (text, box) in enumerate(dataList, 1):
            resultList.append({
                "id": id,
                "polygon": [(int(x), int(y)) for x, y in box.tolist()],
                "text": text,
                "match": bool(self.searchText) and self._checkMatch(text)
            })

        if self.dataType == "data":
            print(self.dataType, json.dumps(resultList, ensure_ascii=False), flush=True)
        
        pilImage.save(f"{PATH_ROOT}{PATH_FILE}output/engine_realtime/{self.uniqueId}/export/{self.fileNameSplit}_result.pdf", format="PDF")

        with open(f"{PATH_ROOT}{PATH_FILE}output/engine_realtime/{self.uniqueId}/export/{self.fileNameSplit}_result.json", "w", encoding="utf-8") as file:
            json.dump(resultList, file, ensure_ascii=False, indent=2)

    def _loadModel(self):
        idBackend = cv2.dnn.DNN_BACKEND_OPENCV
        idTarget = cv2.dnn.DNN_TARGET_CPU

        detector = DetectorPPocr(
            modelPath=f"{PATH_ROOT}src/library/engine_realtime/text_detection_en_ppocrv3_2023may.onnx",
            inputSize=[736, 736],
            binaryThreshold=0.3,
            polygonThreshold=0.5,
            maxCandidates=200,
            unclipRatio=2.0,
            backendId=idBackend,
            targetId=idTarget
        )

        recognizer = RecognizerCrnn(
            modelPath=f"{PATH_ROOT}src/library/engine_realtime/text_recognition_CRNN_EN_2021sep.onnx",
            backendId=idBackend,
            targetId=idTarget
        )

        return detector, recognizer

    def _createOutputDir(self):
        os.makedirs(f"{PATH_ROOT}{PATH_FILE}output/engine_realtime/{self.uniqueId}/layout/", exist_ok=True)
        os.makedirs(f"{PATH_ROOT}{PATH_FILE}output/engine_realtime/{self.uniqueId}/export/", exist_ok=True)

    def _execute(self, languageValue="", fileNameValue="", uniqueIdValue="", searchTextValue="", dataTypeValue="file"):
        self.language = languageValue
        self.fileName = fileNameValue
        self.fileNameSplit = ".".join(self.fileName.split(".")[:-1])
        self.uniqueId = uniqueIdValue
        self.searchText = searchTextValue
        self.dataType = dataTypeValue

        self._createOutputDir()
        
        detector, recognizer = self._loadModel()

        self._inference(detector, recognizer)

        if self.dataType == "file":
            print(self.dataType, flush=True)

    def __init__(self):
        self.language = ""
        self.fileName = ""
        self.uniqueId = ""
        self.searchText = ""
        self.dataType = ""
        self.fontName = "NotoSansJP-Regular.ttf"
        self.argumentCaseSensitive = False
        self.argumentSelective = False
        self.argumentContain = True
