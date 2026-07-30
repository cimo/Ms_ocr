import sys
sys.dont_write_bytecode = True

import os
import io
import math
import cv2
import numpy

sys.path.append(f"{os.path.dirname(__file__)}/..")
from helper import onnxSessionBuild

class Recognition:
    def _imageCrop(self, image, coordinateList):
        pointList = numpy.array(coordinateList, dtype=numpy.float32)

        widthCrop = int(max(numpy.linalg.norm(pointList[0] - pointList[1]), numpy.linalg.norm(pointList[2] - pointList[3])))
        heightCrop = int(max(numpy.linalg.norm(pointList[0] - pointList[3]), numpy.linalg.norm(pointList[1] - pointList[2])))

        if widthCrop < 1 or heightCrop < 1:
            return None

        pointTargetList = numpy.array([
            [0, 0],
            [widthCrop, 0],
            [widthCrop, heightCrop],
            [0, heightCrop]
        ], dtype=numpy.float32)

        matrix = cv2.getPerspectiveTransform(pointList, pointTargetList)

        imageCrop = cv2.warpPerspective(
            image,
            matrix,
            (widthCrop, heightCrop),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC
        )

        if heightCrop / float(widthCrop) >= self.ratioRotate:
            imageCrop = numpy.ascontiguousarray(numpy.rot90(imageCrop))

        return imageCrop

    def _imageResize(self, image):
        imageHeight, imageWidth = image.shape[0:2]

        ratioWidthHeight = max(self.imageWidthModel / float(self.imageHeightModel), imageWidth / float(imageHeight))

        widthTarget = int(self.imageHeightModel * ratioWidthHeight)

        if widthTarget > self.imageWidthMax:
            widthTarget = self.imageWidthMax
            widthResized = self.imageWidthMax
        else:
            widthResized = int(math.ceil(self.imageHeightModel * imageWidth / float(imageHeight)))

            if widthResized > widthTarget:
                widthResized = widthTarget

        imageResized = cv2.resize(image, (widthResized, self.imageHeightModel))

        tensor = imageResized.astype(numpy.float32).transpose((2, 0, 1)) / 255.0
        tensor = (tensor - 0.5) / 0.5

        tensorPadded = numpy.zeros((3, self.imageHeightModel, widthTarget), dtype=numpy.float32)
        tensorPadded[:, :, 0:widthResized] = tensor

        return numpy.expand_dims(tensorPadded, axis=0)

    def execute(self, image, coordinateList):
        imageCrop = self._imageCrop(image, coordinateList)

        if imageCrop is None:
            return {"text": "", "score": 0.0}

        tensor = self._imageResize(imageCrop)

        tensorOutputList = self.onnxSession.run(None, {"x": tensor})

        probability = tensorOutputList[0][0]

        indexList = probability.argmax(axis=-1)
        valueList = probability.max(axis=-1)

        text = ""
        scoreList = []

        for a in range(len(indexList)):
            if indexList[a] == 0:
                continue

            if a > 0 and indexList[a] == indexList[a - 1]:
                continue

            text += self.characterList[indexList[a]]

            scoreList.append(float(valueList[a]))

        score = 0.0

        if len(scoreList) > 0:
            score = float(numpy.mean(scoreList))

        return {
            "text": text,
            "score": score
        }

    def __init__(self):
        self.osPathDirName = f"{os.path.dirname(__file__)}/"
        self.pathModel = f"{self.osPathDirName}model/pp-ocrV6_medium_rec.onnx"
        self.pathDictionary = f"{self.osPathDirName}model/dictionary.txt"

        self.imageHeightModel = 48
        self.imageWidthModel = 320
        self.imageWidthMax = 3200
        self.ratioRotate = 1.5

        self.characterList = ["blank"]

        lineList = io.open(self.pathDictionary, encoding="utf-8").read().split("\n")

        for a in range(len(lineList)):
            if lineList[a] != "":
                self.characterList.append(lineList[a])

        self.characterList.append(" ")

        self.onnxSession = onnxSessionBuild(self.pathModel)
