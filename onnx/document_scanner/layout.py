import sys
import os
import cv2
import numpy
import json

sys.dont_write_bytecode = True
sys.path.append(f"{os.path.dirname(__file__)}/..")

# Source
from helper import onnxSessionBuild

class Layout:
    def _detect(self, image):
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

            label = self.labelObject[classId] if classId in self.labelObject else str(classId)

            if score < (self.scoreThresholdObject[label] if label in self.scoreThresholdObject else self.scoreThreshold):
                continue

            x1 = max(0, min(int(round(float(value[2]))), imageWidth))
            y1 = max(0, min(int(round(float(value[3]))), imageHeight))
            x2 = max(0, min(int(round(float(value[4]))), imageWidth))
            y2 = max(0, min(int(round(float(value[5]))), imageHeight))

            if x2 <= x1 or y2 <= y1:
                continue

            resultList.append({
                "label": label,
                "score": score,
                "bbox": [x1, y1, x2, y2],
                "centerPoint": self._centerPointCalculate([x1, y1, x2, y2]),
                "path": ""
            })

        return self._boxContainedRemove(self._boxSuppression(resultList))

    def _centerPointCalculate(self, bboxList):
        return {
            "x": int(round((bboxList[0] + bboxList[2]) / 2)),
            "y": int(round((bboxList[1] + bboxList[3]) / 2))
        }

    def _boxSuppression(self, boxList):
        resultList = []

        boxSortedList = sorted(boxList, key=lambda boxObject: (boxObject["label"] in self.labelContainerList, boxObject["score"]), reverse=True)

        for a in range(len(boxSortedList)):
            bboxList = boxSortedList[a]["bbox"]

            area = (bboxList[2] - bboxList[0]) * (bboxList[3] - bboxList[1])

            isKeep = True

            for b in range(len(resultList)):
                bboxKeptList = resultList[b]["bbox"]

                areaKept = (bboxKeptList[2] - bboxKeptList[0]) * (bboxKeptList[3] - bboxKeptList[1])

                x1 = max(bboxList[0], bboxKeptList[0])
                y1 = max(bboxList[1], bboxKeptList[1])
                x2 = min(bboxList[2], bboxKeptList[2])
                y2 = min(bboxList[3], bboxKeptList[3])

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

    def _labelGroupGet(self, label):
        if label in self.labelGroupObject:
            return self.labelGroupObject[label]

        return label

    def _boxContainedRemove(self, boxList):
        resultList = []

        for a in range(len(boxList)):
            bboxList = boxList[a]["bbox"]

            area = (bboxList[2] - bboxList[0]) * (bboxList[3] - bboxList[1])

            isContained = False

            for b in range(len(boxList)):
                if a == b:
                    continue

                bboxParentList = boxList[b]["bbox"]

                areaParent = (bboxParentList[2] - bboxParentList[0]) * (bboxParentList[3] - bboxParentList[1])

                if areaParent <= area:
                    continue

                x1 = max(bboxList[0], bboxParentList[0])
                y1 = max(bboxList[1], bboxParentList[1])
                x2 = min(bboxList[2], bboxParentList[2])
                y2 = min(bboxList[3], bboxParentList[3])

                if x2 <= x1 or y2 <= y1:
                    continue

                if (x2 - x1) * (y2 - y1) / float(area) >= self.levelBoxContained:
                    isContained = True

                    break

            if isContained == False:
                resultList.append(boxList[a])

        return resultList

    def _itemOrder(self, itemList, imageWidth):
        itemHeaderList = []
        itemFooterList = []
        itemBodyList = []

        for a in range(len(itemList)):
            if itemList[a]["label"] in self.labelHeaderList:
                itemHeaderList.append(itemList[a])
            elif itemList[a]["label"] in self.labelFooterList:
                itemFooterList.append(itemList[a])
            else:
                itemBodyList.append(itemList[a])

        resultList = sorted(itemHeaderList, key=lambda itemObject: itemObject["bbox"][1])

        itemSortedList = sorted(itemBodyList, key=lambda itemObject: itemObject["bbox"][1])

        itemBandList = []

        for a in range(len(itemSortedList)):
            bboxList = itemSortedList[a]["bbox"]

            if (bboxList[2] - bboxList[0]) / float(imageWidth) < self.levelFullWidth:
                itemBandList.append(itemSortedList[a])

                continue

            resultList = resultList + self._bandOrder(itemBandList)
            resultList.append(itemSortedList[a])

            itemBandList = []

        resultList = resultList + self._bandOrder(itemBandList)

        return resultList + sorted(itemFooterList, key=lambda itemObject: itemObject["bbox"][1])

    def _bandOrder(self, itemList):
        resultList = []

        columnList = self._columnGroup(itemList)

        for a in range(len(columnList)):
            itemColumnList = sorted(columnList[a]["itemList"], key=lambda itemObject: itemObject["bbox"][1])

            for b in range(len(itemColumnList)):
                resultList.append(itemColumnList[b])

        return resultList

    def _columnGroup(self, itemList):
        resultList = []

        itemSortedList = sorted(itemList, key=lambda itemObject: itemObject["bbox"][0])

        for a in range(len(itemSortedList)):
            bboxList = itemSortedList[a]["bbox"]

            isAdded = False

            for b in range(len(resultList)):
                x1 = max(bboxList[0], resultList[b]["x1"])
                x2 = min(bboxList[2], resultList[b]["x2"])

                if x2 <= x1:
                    continue

                if (x2 - x1) / float(min(bboxList[2] - bboxList[0], resultList[b]["x2"] - resultList[b]["x1"])) < self.levelColumnOverlap:
                    continue

                resultList[b]["x1"] = min(resultList[b]["x1"], bboxList[0])
                resultList[b]["x2"] = max(resultList[b]["x2"], bboxList[2])

                resultList[b]["itemList"].append(itemSortedList[a])

                isAdded = True

                break

            if isAdded == False:
                resultList.append({"x1": bboxList[0], "x2": bboxList[2], "itemList": [itemSortedList[a]]})

        return sorted(resultList, key=lambda columnObject: columnObject["x1"])

    def _itemFlow(self, itemObject, itemList):
        if itemObject["label"] not in self.labelSecondaryList:
            return "main"

        if itemObject["label"] in self.labelFigureTitleList and self._figureNear(itemList, itemObject["bbox"]) == False:
            return "main"

        return "secondary"

    def _figureNear(self, itemList, bboxList):
        for a in range(len(itemList)):
            if itemList[a]["label"] not in self.labelFigureList:
                continue

            bboxFigureList = itemList[a]["bbox"]

            x1 = max(bboxList[0], bboxFigureList[0])
            x2 = min(bboxList[2], bboxFigureList[2])

            if x2 <= x1:
                continue

            if bboxList[1] - bboxFigureList[3] < 0 or bboxList[1] - bboxFigureList[3] > (bboxList[3] - bboxList[1]) * self.levelFigureGap:
                continue

            return True

        return False

    def _mediaWrite(self, itemList, image, numberPage, pathOutput):
        for a in range(len(itemList)):
            if itemList[a]["label"] not in self.labelFigureList:
                continue

            bboxList = itemList[a]["bbox"]

            imageCrop = image[bboxList[1]:bboxList[3], bboxList[0]:bboxList[2]]

            fileName = f"{numberPage}_{a + 1}.jpg"

            os.makedirs(f"{pathOutput}media/", exist_ok=True)

            cv2.imwrite(f"{pathOutput}media/{fileName}", imageCrop)

            itemList[a]["path"] = f"media/{fileName}"

    def _debugBox(self, image, itemList, pathOutput, numberPage):
        imageDebug = image.copy()

        labelDrawnList = []

        for a in range(len(itemList)):
            bboxList = itemList[a]["bbox"]

            color = self.labelColorObject[itemList[a]["label"]] if itemList[a]["label"] in self.labelColorObject else (0, 0, 0)

            boxRegion = imageDebug[bboxList[1]:bboxList[3], bboxList[0]:bboxList[2]]
            boxOverlay = numpy.full(boxRegion.shape, color, dtype=numpy.uint8)

            imageDebug[bboxList[1]:bboxList[3], bboxList[0]:bboxList[2]] = cv2.addWeighted(boxOverlay, self.levelDebugOpacity, boxRegion, 1 - self.levelDebugOpacity, 0)

            text = f"{itemList[a]['label']} {itemList[a]['score']:.2f}"

            textWidth, textHeight = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

            x = bboxList[0]
            y = max(textHeight, bboxList[1] - 6)

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

    def resultBuild(self, astPage, countStart):
        resultList = []

        flowObject = {"main": astPage["itemMainList"], "secondary": astPage["itemSecondaryList"]}

        for flow in flowObject:
            itemList = flowObject[flow]

            for a in range(len(itemList)):
                resultList.append({
                    "id": countStart + len(resultList) + 1,
                    "page": astPage["number"],
                    "flow": flow,
                    "label": itemList[a]["label"],
                    "score": itemList[a]["score"],
                    "bbox": itemList[a]["bbox"],
                    "centerPoint": itemList[a]["centerPoint"],
                    "path": itemList[a]["path"]
                })

        return resultList

    def astWrite(self, pathOutput, astPageList):
        with open(f"{pathOutput}debug/layout/ast.json", "w", encoding="utf-8") as file:
            json.dump({"pageList": astPageList}, file, ensure_ascii=False, indent=4)

    def execute(self, pathOutput, image, numberPage):
        imageHeight, imageWidth = image.shape[0:2]

        itemList = self._itemOrder(self._detect(image), imageWidth)

        itemMainList = []
        itemSecondaryList = []

        for a in range(len(itemList)):
            if self._itemFlow(itemList[a], itemList) == "main":
                itemMainList.append(itemList[a])
            else:
                itemSecondaryList.append(itemList[a])

        self._mediaWrite(itemList, image, numberPage, pathOutput)
        self._debugBox(image, itemList, pathOutput, numberPage)

        return {"number": numberPage, "width": imageWidth, "height": imageHeight, "itemMainList": itemMainList, "itemSecondaryList": itemSecondaryList}

    def __init__(self):
        self.osPathDirName = f"{os.path.dirname(__file__)}/"
        self.pathModelLayout = f"{self.osPathDirName}model/pp-docLayout_plus-l.onnx"

        self.imageSizeLayout = 800

        self.levelBoxContained = 0.9
        self.levelBoxNms = 0.5
        self.levelColumnOverlap = 0.5
        self.levelFigureGap = 2.0
        self.levelFullWidth = 0.7
        self.levelDebugOpacity = 0.2

        self.scoreThreshold = 0.3
        self.scoreThresholdObject = {"table": 0.35}

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

        self.labelContainerList = ["table", "image", "chart"]
        self.labelFigureList = ["image", "chart"]
        self.labelFigureTitleList = ["figure_title"]
        self.labelHeaderList = ["header"]
        self.labelFooterList = ["footer"]
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

        self.onnxSessionLayout = onnxSessionBuild(self.pathModelLayout)
