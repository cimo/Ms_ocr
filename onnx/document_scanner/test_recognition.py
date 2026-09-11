import sys
import os
import cv2
import numpy
import io
import math

sys.dont_write_bytecode = True
sys.path.append(f"{os.path.dirname(__file__)}/..")

# Source
from helper import onnxSessionBuild

class Recognition:
    def _imageCrop(self, coordinateList, image):
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

    def _textDecode(self, probability):
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

    def _batchGroup(self, indexList, imageCropList):
        resultList = []

        groupList = []
        ratioStart = 0.0

        for a in range(len(indexList)):
            ratio = imageCropList[indexList[a]].shape[1] / float(imageCropList[indexList[a]].shape[0])

            if len(groupList) >= self.sizeBatch or (len(groupList) > 0 and ratio > ratioStart * self.levelBatchRatio):
                resultList.append(groupList)

                groupList = []

            if len(groupList) == 0:
                ratioStart = ratio

            groupList.append(indexList[a])

        if len(groupList) > 0:
            resultList.append(groupList)

        return resultList

    def execute(self, coordinateItemList, image):
        imageCropList = []

        for a in range(len(coordinateItemList)):
            imageCropList.append(self._imageCrop(coordinateItemList[a], image))

        indexList = []

        resultList = []

        for a in range(len(imageCropList)):
            resultList.append({"text": "", "score": 0.0})

            if imageCropList[a] is not None:
                indexList.append(a)

        indexList.sort(key=lambda index: imageCropList[index].shape[1] / float(imageCropList[index].shape[0]))

        groupPageList = self._batchGroup(indexList, imageCropList)

        for a in range(len(groupPageList)):
            groupList = groupPageList[a]

            widthMax = self.imageWidthModel

            imageResizedList = []

            for b in range(len(groupList)):
                imageCrop = imageCropList[groupList[b]]

                widthResized = min(int(math.ceil(self.imageHeightModel * imageCrop.shape[1] / float(imageCrop.shape[0]))), self.imageWidthMax)

                widthMax = max(widthMax, widthResized)

                imageResizedList.append(cv2.resize(imageCrop, (widthResized, self.imageHeightModel)))

            tensorBatch = numpy.zeros((len(imageResizedList), 3, self.imageHeightModel, widthMax), dtype=numpy.float32)

            for b in range(len(imageResizedList)):
                tensor = imageResizedList[b].astype(numpy.float32).transpose((2, 0, 1)) / 255.0
                tensor = (tensor - 0.5) / 0.5

                tensorBatch[b, :, :, 0:tensor.shape[2]] = tensor

            tensorOutputList = self.onnxSession.run(None, {"x": tensorBatch})

            for b in range(len(groupList)):
                resultList[groupList[b]] = self._textDecode(tensorOutputList[0][b])

        return resultList

    def __init__(self):
        self.osPathDirName = f"{os.path.dirname(__file__)}/"
        self.pathModel = f"{self.osPathDirName}model/pp-ocrV6_medium_rec.onnx"
        self.pathDictionary = f"{self.osPathDirName}model/dictionary.txt"

        self.imageHeightModel = 48
        self.imageWidthModel = 320
        self.imageWidthMax = 3200
        
        self.ratioRotate = 1.5
        self.sizeBatch = 16
        self.levelBatchRatio = 1.25

        self.characterList = ["blank"]

        lineList = io.open(self.pathDictionary, encoding="utf-8").read().split("\n")

        for a in range(len(lineList)):
            if lineList[a] != "":
                self.characterList.append(lineList[a])

        self.characterList.append(" ")

        self.onnxSession = onnxSessionBuild(self.pathModel)
