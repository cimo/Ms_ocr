import sys
import os
import cv2
import numpy
import json

sys.dont_write_bytecode = True
sys.path.append(f"{os.path.dirname(__file__)}/..")

# Source
from helper import onnxSessionBuild

class Test:
    def _layoutDetect(self, image):
        imageHeight, imageWidth = image.shape[0:2]

        imageRgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        imageResized = cv2.resize(imageRgb, (self.imageSizeLayout, self.imageSizeLayout), interpolation=cv2.INTER_CUBIC).astype(numpy.float32) / 255.0

        tensor = numpy.expand_dims(imageResized.transpose((2, 0, 1)), axis=0).astype(numpy.float32)

        tensorFeedObject = {
            "image": tensor,
            "im_shape": numpy.array([[self.imageSizeLayout, self.imageSizeLayout]], dtype=numpy.float32),
            "scale_factor": numpy.array([[self.imageSizeLayout / float(imageHeight), self.imageSizeLayout / float(imageWidth)]], dtype=numpy.float32)
        }

        tensorOutputList = self.onnxSessionLayout.run(None, tensorFeedObject)

        boxCount = int(tensorOutputList[1][0])

        resultList = []

        for a in range(boxCount):
            value = tensorOutputList[0][a]

            classId = int(value[0])
            score = float(value[1])

            if score < self.scoreThreshold:
                continue

            x1 = max(0, min(int(round(float(value[2]))), imageWidth))
            y1 = max(0, min(int(round(float(value[3]))), imageHeight))
            x2 = max(0, min(int(round(float(value[4]))), imageWidth))
            y2 = max(0, min(int(round(float(value[5]))), imageHeight))

            if x2 <= x1 or y2 <= y1:
                continue

            resultList.append({
                "label": self.labelObject[classId] if classId in self.labelObject else str(classId),
                "score": score,
                "coordinate": [x1, y1, x2, y2]
            })

        return self._boxContainedRemove(self._boxSuppression(resultList))

    def _itemFlow(self, label):
        if label in self.labelSecondaryList:
            return "secondary"

        return "main"

    def _labelGroupGet(self, label):
        if label in self.labelGroupObject:
            return self.labelGroupObject[label]

        return label

    def _boxSuppression(self, boxList):
        resultList = []

        boxSortedList = sorted(boxList, key=lambda boxObject: (boxObject["label"] in self.labelContainerList, boxObject["score"]), reverse=True)

        for a in range(len(boxSortedList)):
            coordinateList = boxSortedList[a]["coordinate"]

            area = (coordinateList[2] - coordinateList[0]) * (coordinateList[3] - coordinateList[1])

            isKeep = True

            for b in range(len(resultList)):
                coordinateKeptList = resultList[b]["coordinate"]

                areaKept = (coordinateKeptList[2] - coordinateKeptList[0]) * (coordinateKeptList[3] - coordinateKeptList[1])

                x1 = max(coordinateList[0], coordinateKeptList[0])
                y1 = max(coordinateList[1], coordinateKeptList[1])
                x2 = min(coordinateList[2], coordinateKeptList[2])
                y2 = min(coordinateList[3], coordinateKeptList[3])

                if x2 <= x1 or y2 <= y1:
                    continue

                areaIntersection = (x2 - x1) * (y2 - y1)

                isSameRegion = areaIntersection / float(area + areaKept - areaIntersection) >= self.levelBoxNms
                isInside = areaIntersection / float(min(area, areaKept)) >= self.levelBoxContained
                isSameGroup = self._labelGroupGet(boxSortedList[a]["label"]) == self._labelGroupGet(resultList[b]["label"])

                if isSameRegion or (isInside and isSameGroup):
                    isKeep = False

                    break

            if isKeep:
                resultList.append(boxSortedList[a])

        return resultList

    def _boxContainedRemove(self, boxList):
        resultList = []

        for a in range(len(boxList)):
            coordinateList = boxList[a]["coordinate"]

            area = (coordinateList[2] - coordinateList[0]) * (coordinateList[3] - coordinateList[1])

            isContained = False

            for b in range(len(boxList)):
                if a == b:
                    continue

                coordinateParentList = boxList[b]["coordinate"]

                areaParent = (coordinateParentList[2] - coordinateParentList[0]) * (coordinateParentList[3] - coordinateParentList[1])

                if areaParent <= area:
                    continue

                x1 = max(coordinateList[0], coordinateParentList[0])
                y1 = max(coordinateList[1], coordinateParentList[1])
                x2 = min(coordinateList[2], coordinateParentList[2])
                y2 = min(coordinateList[3], coordinateParentList[3])

                if x2 <= x1 or y2 <= y1:
                    continue

                if (x2 - x1) * (y2 - y1) / float(area) >= self.levelBoxContained:
                    isContained = True

                    break

            if isContained == False:
                resultList.append(boxList[a])

        return resultList

    def _debugLayout(self, image, layoutList, pathOutput, numberPage):
        imageDebug = image.copy()

        labelDrawnList = []

        for a in range(len(layoutList)):
            coordinateList = layoutList[a]["coordinate"]

            color = self.labelColorObject[layoutList[a]["label"]] if layoutList[a]["label"] in self.labelColorObject else self.colorLabel

            boxRegion = imageDebug[coordinateList[1]:coordinateList[3], coordinateList[0]:coordinateList[2]]
            boxOverlay = numpy.full(boxRegion.shape, color, dtype=numpy.uint8)

            imageDebug[coordinateList[1]:coordinateList[3], coordinateList[0]:coordinateList[2]] = cv2.addWeighted(boxOverlay, self.levelDebugOpacity, boxRegion, 1 - self.levelDebugOpacity, 0)

            text = f"{layoutList[a]['label']} {layoutList[a]['score']:.2f}"

            textWidth, textHeight = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

            x = coordinateList[0]
            y = max(textHeight, coordinateList[1] - 6)

            isOverlap = True

            while isOverlap:
                isOverlap = False

                for b in range(len(labelDrawnList)):
                    isSameRow = abs(y - labelDrawnList[b]["y"]) < textHeight + 4
                    isSameColumn = x < labelDrawnList[b]["x"] + labelDrawnList[b]["width"] and labelDrawnList[b]["x"] < x + textWidth

                    if isSameRow and isSameColumn:
                        isOverlap = True
                        y = labelDrawnList[b]["y"] - textHeight - 4

                        break

            labelDrawnList.append({"x": x, "y": y, "width": textWidth})

            cv2.putText(imageDebug, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imwrite(f"{pathOutput}debug/layout/{numberPage}.jpg", imageDebug)

    def _astWrite(self, pathOutput):
        with open(f"{pathOutput}debug/layout/{self.astFileName}", "w", encoding="utf-8") as file:
            json.dump({"pageList": self.astPageList}, file, ensure_ascii=False, indent=4)

    def execute(self, image, numberPage, pathOutput):
        if pathOutput != self.pathOutputCurrent:
            self.pathOutputCurrent = pathOutput

            self.astPageList = []

        imageHeight, imageWidth = image.shape[0:2]

        itemList = self._layoutDetect(image)

        itemMainList = []
        itemSecondaryList = []

        for a in range(len(itemList)):
            if self._itemFlow(itemList[a]["label"]) == "main":
                itemMainList.append(itemList[a])
            else:
                itemSecondaryList.append(itemList[a])

        astPage = {"number": numberPage, "width": imageWidth, "height": imageHeight, "itemMainList": itemMainList, "itemSecondaryList": itemSecondaryList}

        self.astPageList.append(astPage)

        self._debugLayout(image, itemList, pathOutput, numberPage)
        self._astWrite(pathOutput)

        return astPage

    def __init__(self):
        self.osPathDirName = f"{os.path.dirname(__file__)}/"
        self.pathModelLayout = f"{self.osPathDirName}model/pp-docLayout_plus-l.onnx"

        self.astFileName = "ast.json"

        self.astPageList = []

        self.pathOutputCurrent = ""

        self.imageSizeLayout = 800

        self.levelBoxContained = 0.9
        self.levelBoxNms = 0.5
        self.levelDebugOpacity = 0.2

        self.scoreThreshold = 0.3

        self.colorLabel = (0, 0, 0)

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

        self.labelContainerList = ["table", "image", "chart"]

        self.labelSecondaryList = [
            "image",
            "figure_title",
            "chart",
            "formula",
            "formula_number",
            "algorithm",
            "aside_text",
            "footnote",
            "seal"
        ]

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

        self.labelColorObject = {
            "header": (128, 128, 128),
            "doc_title": (255, 0, 0),
            "abstract": (0, 200, 0),
            "content": (0, 200, 0),
            "paragraph_title": (255, 0, 0),
            "text": (0, 200, 0),
            "image": (0, 165, 255),
            "figure_title": (255, 0, 0),
            "chart": (0, 165, 255),
            "table": (0, 0, 255),
            "formula": (200, 0, 200),
            "formula_number": (200, 0, 200),
            "algorithm": (0, 200, 0),
            "aside_text": (128, 128, 128),
            "reference": (0, 200, 0),
            "reference_content": (0, 200, 0),
            "footnote": (128, 128, 128),
            "footer": (128, 128, 128),
            "number": (128, 128, 128),
            "seal": (128, 128, 128)
        }

        self.onnxSessionLayout = onnxSessionBuild(self.pathModelLayout)
