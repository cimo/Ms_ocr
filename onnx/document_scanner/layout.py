import sys
sys.dont_write_bytecode = True

import os
import cv2
import numpy

sys.path.append(f"{os.path.dirname(__file__)}/..")
from helper import onnxSessionBuild

class Layout:
    def _overlapCalculate(self, coordinateFirstList, coordinateSecondList):
        areaFirst = (coordinateFirstList[2] - coordinateFirstList[0]) * (coordinateFirstList[3] - coordinateFirstList[1])
        areaSecond = (coordinateSecondList[2] - coordinateSecondList[0]) * (coordinateSecondList[3] - coordinateSecondList[1])

        x1 = max(coordinateFirstList[0], coordinateSecondList[0])
        y1 = max(coordinateFirstList[1], coordinateSecondList[1])
        x2 = min(coordinateFirstList[2], coordinateSecondList[2])
        y2 = min(coordinateFirstList[3], coordinateSecondList[3])

        if x2 <= x1 or y2 <= y1:
            return {"intersectionOverUnion": 0.0, "containment": 0.0, "areaFirst": areaFirst, "areaSecond": areaSecond}

        areaIntersection = (x2 - x1) * (y2 - y1)

        return {
            "intersectionOverUnion": areaIntersection / float(areaFirst + areaSecond - areaIntersection),
            "containment": areaIntersection / float(min(areaFirst, areaSecond)),
            "areaFirst": areaFirst,
            "areaSecond": areaSecond
        }

    def _labelGroupGet(self, label):
        if label in self.labelGroupObject:
            return self.labelGroupObject[label]

        return label

    def _suppressionNonMaximum(self, boxList):
        resultList = []

        boxSortedList = sorted(boxList, key=lambda boxObject: boxObject["score"], reverse=True)

        for a in range(len(boxSortedList)):
            isKeep = True

            for b in range(len(resultList)):
                overlapObject = self._overlapCalculate(boxSortedList[a]["coordinate"], resultList[b]["coordinate"])

                isSameRegion = overlapObject["intersectionOverUnion"] >= self.nmsThreshold
                isInside = overlapObject["containment"] >= self.containmentThreshold
                isSameGroup = self._labelGroupGet(boxSortedList[a]["label"]) == self._labelGroupGet(resultList[b]["label"])

                if isSameRegion or (isInside and isSameGroup):
                    isKeep = False

                    break

            if isKeep:
                resultList.append(boxSortedList[a])

        return resultList

    def _suppressionContainer(self, boxList):
        resultList = []

        for a in range(len(boxList)):
            isKeep = True

            for b in range(len(boxList)):
                if a == b or boxList[b]["label"] not in self.labelContainerList:
                    continue

                overlapObject = self._overlapCalculate(boxList[a]["coordinate"], boxList[b]["coordinate"])

                if overlapObject["containment"] >= self.containmentThreshold and overlapObject["areaFirst"] < overlapObject["areaSecond"]:
                    isKeep = False

                    break

            if isKeep:
                resultList.append(boxList[a])

        return resultList

    def execute(self, imageRgb):
        resultList = []

        imageHeight, imageWidth = imageRgb.shape[0:2]
        imageResized = cv2.resize(imageRgb, (800, 800), interpolation=cv2.INTER_CUBIC).astype(numpy.float32) / 255.0

        tensor = numpy.expand_dims(imageResized.transpose((2, 0, 1)), axis=0).astype(numpy.float32)

        tensorFeedObject = {
            "image": tensor,
            "im_shape": numpy.array([[800, 800]], dtype=numpy.float32),
            "scale_factor": numpy.array([[800 / float(imageHeight), 800 / float(imageWidth)]], dtype=numpy.float32)
        }

        tensorOutputList = self.onnxSession.run(None, tensorFeedObject)

        boxCount = int(tensorOutputList[1][0]) if len(tensorOutputList) > 1 else len(tensorOutputList[0])

        for a in range(boxCount):
            value = tensorOutputList[0][a]

            classId = int(value[0])
            score = float(value[1])
            x1 = max(0, min(int(round(float(value[2]))), imageWidth))
            y1 = max(0, min(int(round(float(value[3]))), imageHeight))
            x2 = max(0, min(int(round(float(value[4]))), imageWidth))
            y2 = max(0, min(int(round(float(value[5]))), imageHeight))

            if score >= self.scoreThreshold and x2 > x1 and y2 > y1:
                label = self.labelObject[classId] if classId in self.labelObject else str(classId)

                resultList.append({
                    "label": label,
                    "score": score,
                    "coordinate": [x1, y1, x2, y2]
                })

        return self._suppressionContainer(self._suppressionNonMaximum(resultList))

    def __init__(self):
        self.osPathDirName = f"{os.path.dirname(__file__)}/"
        self.pathModel = f"{self.osPathDirName}model/pp-docLayout_plus-l.onnx"

        self.scoreThreshold = 0.3
        self.nmsThreshold = 0.5
        self.containmentThreshold = 0.9

        self.labelGroupObject = {
            "text": "text",
            "header": "text",
            "footer": "text",
            "footnote": "text",
            "aside_text": "text",
            "abstract": "text",
            "content": "text",
            "reference_content": "text",
            "paragraph_title": "text",
            "figure_title": "text"
        }

        self.labelContainerList = ["table", "image", "chart"]

        self.labelObject = {
            12: "header",
            10: "doc_title",
            4: "abstract",
            5: "content",
            0: "paragraph_title",
            2: "text",
            1: "image",
            6: "figure_title",
            16: "chart",
            8: "table",
            7: "formula",
            17: "formula_number",
            13: "algorithm",
            18: "aside_text",
            9: "reference",
            19: "reference_content",
            11: "footnote",
            14: "footer",
            3: "number",
            15: "seal"
        }

        self.onnxSession = onnxSessionBuild(self.pathModel)
