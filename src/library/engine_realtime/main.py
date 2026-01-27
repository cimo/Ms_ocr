import os
import logging
import ast
import cv2
import numpy
import re
import unicodedata

# Source
from .ppocr_detection import PPocrDetection
from .crnn_recognition import CrnnRecognition

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
    def _normalizeText(self, value):
        if value is None:
            return ""
        
        value = unicodedata.normalize("NFKC", value).strip()

        return value.lower() if self.argumentCaseSensitive else value

    def _checkMatch(self, text):
        textNormalize = self._normalizeText(text)
        matchNormalize = self._normalizeText(self.matchText)

        if self.argumentRegex:
            flag = re.IGNORECASE if self.argumentCaseSensitive else 0

            return re.search(self.matchText, text, flag) is not None
        
        if self.argumentContain:
            return matchNormalize in textNormalize
        
        return textNormalize == matchNormalize

    def _inference(self, imageRgb, detector, recognizer):
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

        red = (0, 0, 255)
        green = (0, 200, 0)

        matchList = []

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

            if self.matchText is not None and self._checkMatch(text):
                color = green

                matchList.append((text, box.copy()))
            else:
                color = red

            cv2.polylines(left, [box], isClosed=True, color=color, thickness=1)
            cv2.polylines(right, [box], isClosed=True, color=color, thickness=1)

            x0, y0 = box[0]
            textY = max(0, int(y0 - 5))

            cv2.putText(right, text, (int(x0), textY), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 0), fontThickness, lineType=cv2.LINE_AA)

        return cv2.hconcat([left, right]), matchList

    def _process(self, detector, recognizer):
        imageRead = cv2.imread(f"{PATH_ROOT}{PATH_FILE}input/{self.fileName}")

        if imageRead is None:
            print(f"Error: Cannot read image: {self.fileName}")

            return None

        imageRgb = cv2.cvtColor(imageRead, cv2.COLOR_BGR2RGB)

        output, match = self._inference(imageRgb, detector, recognizer)
        
        pathOutput = f"{PATH_ROOT}{PATH_FILE}output/engine_realtime/"
        os.makedirs(pathOutput, exist_ok=True)

        pathOutputResult = os.path.join(pathOutput, self.fileName)

        cv2.imwrite(pathOutputResult, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

        if self.matchText is not None:
            if IS_DEBUG and match:
                print(f"Match: '{self.matchText}' found {len(match)} in {self.fileName}:")

                for a, (text, box) in enumerate(match, 1):
                    pointList = [(int(x), int(y)) for x, y in box.tolist()]

                    print(f"  {a}. text='{text}'  polygon={pointList}")

        return pathOutputResult

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

    def _execute(self, languageValue, fileNameValue, uniqueIdValue, matchTextValue):
        self.language = languageValue
        self.fileName = fileNameValue
        self.fileNameSplit = ".".join(self.fileName.split(".")[:-1])
        self.uniqueId = uniqueIdValue
        self.matchText = matchTextValue
        
        detector, recognizer = self._loadModel()

        self._process(detector, recognizer)

        print("ok", flush=True)

    def __init__(self):
        self.language = ""
        self.fileName = ""
        self.uniqueId = ""
        self.matchText = ""
        self.argumentContain = False
        self.argumentRegex = False
        self.argumentCaseSensitive = True
