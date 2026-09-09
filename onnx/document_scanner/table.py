import sys
import os
import cv2
import numpy

sys.dont_write_bytecode = True
sys.path.append(f"{os.path.dirname(__file__)}/..")

# Source
from helper import onnxSessionBuild

class Cell:
    def _typeClassify(self, imageRgb):
        imageHeight, imageWidth = imageRgb.shape[0:2]

        ratio = self.imageSizeShort / float(min(imageHeight, imageWidth))

        imageResized = cv2.resize(imageRgb, (int(round(imageWidth * ratio)), int(round(imageHeight * ratio))))

        resizedHeight, resizedWidth = imageResized.shape[0:2]

        cropX = int(round((resizedWidth - self.imageSizeCrop) / 2))
        cropY = int(round((resizedHeight - self.imageSizeCrop) / 2))

        imageCrop = imageResized[cropY:cropY + self.imageSizeCrop, cropX:cropX + self.imageSizeCrop]

        tensor = imageCrop.astype(numpy.float32) / 255.0
        tensor = (tensor - self.normalizeMeanList) / self.normalizeStdList

        tensor = numpy.expand_dims(tensor.transpose((2, 0, 1)), axis=0).astype(numpy.float32)

        tensorOutputList = self.onnxSessionClassification.run(None, {"x": tensor})

        return self.labelList[int(numpy.argmax(tensorOutputList[0][0]))]

    def _itemProcess(self, itemRawList, imageWidth, imageHeight):
        resultList = []

        for a in range(len(itemRawList)):
            itemRaw = itemRawList[a]

            score = itemRaw[0]
            x1 = max(0.0, min(itemRaw[1], float(imageWidth)))
            y1 = max(0.0, min(itemRaw[2], float(imageHeight)))
            x2 = max(0.0, min(itemRaw[3], float(imageWidth)))
            y2 = max(0.0, min(itemRaw[4], float(imageHeight)))

            if x2 > x1 and y2 > y1:
                resultList.append({
                    "score": score,
                    "coordinate": [x1, y1, x2, y2]
                })

        return resultList

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

    def _suppressionNonMaximum(self, itemList):
        resultList = []

        itemSortedList = sorted(itemList, key=lambda itemObject: itemObject["score"], reverse=True)

        for a in range(len(itemSortedList)):
            isKeep = True

            for b in range(len(resultList)):
                overlapObject = self._overlapCalculate(itemSortedList[a]["coordinate"], resultList[b]["coordinate"])

                if overlapObject["intersectionOverUnion"] >= self.levelBoxNms:
                    isKeep = False

                    break

            if isKeep:
                resultList.append(itemSortedList[a])

        return resultList

    def _suppressionContainer(self, itemList):
        resultList = []

        for a in range(len(itemList)):
            countContained = 0

            for b in range(len(itemList)):
                if a == b:
                    continue

                overlapObject = self._overlapCalculate(itemList[a]["coordinate"], itemList[b]["coordinate"])

                if overlapObject["containment"] >= self.levelBoxContained and overlapObject["areaFirst"] > overlapObject["areaSecond"]:
                    countContained += 1

            if countContained < self.countContainedMinimum:
                resultList.append(itemList[a])

        return resultList

    def _inference(self, imageRgb, tableType):
        imageHeight, imageWidth = imageRgb.shape[0:2]
        imageResized = cv2.resize(imageRgb, (self.imageSizeDetection, self.imageSizeDetection), interpolation=cv2.INTER_CUBIC).astype(numpy.float32) / 255.0

        tensor = numpy.expand_dims(imageResized.transpose((2, 0, 1)), axis=0).astype(numpy.float32)

        tensorFeedObject = {
            "image": tensor,
            "im_shape": numpy.array([[self.imageSizeDetection, self.imageSizeDetection]], dtype=numpy.float32),
            "scale_factor": numpy.array([[self.imageSizeDetection / float(imageHeight), self.imageSizeDetection / float(imageWidth)]], dtype=numpy.float32)
        }

        tensorOutputList = self.onnxSessionDetectionObject[tableType].run(None, tensorFeedObject)

        boxCount = int(tensorOutputList[1][0]) if len(tensorOutputList) > 1 else len(tensorOutputList[0])

        itemRawList = []

        for a in range(boxCount):
            value = tensorOutputList[0][a]

            score = float(value[1])

            if score >= self.scoreThreshold:
                itemRawList.append([score, float(value[2]), float(value[3]), float(value[4]), float(value[5])])

        itemList = self._itemProcess(itemRawList, imageWidth, imageHeight)
        itemList = self._suppressionNonMaximum(itemList)
        itemList = self._suppressionContainer(itemList)

        return itemList

    def _edgeCluster(self, valueList, tolerance):
        groupList = []

        valueSortedList = sorted(valueList)

        for a in range(len(valueSortedList)):
            if len(groupList) > 0 and valueSortedList[a] - groupList[len(groupList) - 1][-1] <= tolerance:
                groupList[len(groupList) - 1].append(valueSortedList[a])
            else:
                groupList.append([valueSortedList[a]])

        resultList = []

        for a in range(len(groupList)):
            resultList.append({"value": sum(groupList[a]) / len(groupList[a]), "support": len(groupList[a])})

        return resultList

    def _edgeFilter(self, groupList):
        resultList = []

        supportMaximum = 0

        for a in range(len(groupList)):
            supportMaximum = max(supportMaximum, groupList[a]["support"])

        supportMinimum = max(1, int(round(supportMaximum * self.levelEdgeSupport)))

        for a in range(len(groupList)):
            isBorder = a == 0 or a == len(groupList) - 1

            if isBorder or groupList[a]["support"] >= supportMinimum:
                resultList.append(groupList[a]["value"])

        return resultList

    def _indexNearest(self, edgeList, value):
        result = 0

        for a in range(len(edgeList)):
            if abs(value - edgeList[a]) < abs(value - edgeList[result]):
                result = a

        return result

    def _cellMerge(self, cellList):
        slotObject = {}

        for a in range(len(cellList)):
            key = (cellList[a]["rowIndex"], cellList[a]["columnIndex"])

            if key not in slotObject:
                slotObject[key] = cellList[a]
            else:
                cell = slotObject[key]

                cell["score"] = max(cell["score"], cellList[a]["score"])
                cell["coordinate"] = [
                    min(cell["coordinate"][0], cellList[a]["coordinate"][0]),
                    min(cell["coordinate"][1], cellList[a]["coordinate"][1]),
                    max(cell["coordinate"][2], cellList[a]["coordinate"][2]),
                    max(cell["coordinate"][3], cellList[a]["coordinate"][3])
                ]
                cell["columnSpan"] = max(cell["columnSpan"], cellList[a]["columnSpan"])
                cell["rowSpan"] = max(cell["rowSpan"], cellList[a]["rowSpan"])

        resultList = []

        for key in slotObject:
            resultList.append(slotObject[key])

        return resultList

    def _gridBuild(self, cellList, tableWidth, tableHeight):
        edgeXList = []
        edgeYList = []

        for a in range(len(cellList)):
            edgeXList.append(cellList[a]["coordinate"][0])
            edgeXList.append(cellList[a]["coordinate"][2])
            edgeYList.append(cellList[a]["coordinate"][1])
            edgeYList.append(cellList[a]["coordinate"][3])

        edgeXList = self._edgeFilter(self._edgeCluster(edgeXList, tableWidth * self.levelEdgeSnap))
        edgeYList = self._edgeFilter(self._edgeCluster(edgeYList, tableHeight * self.levelEdgeSnap))

        for a in range(len(cellList)):
            columnStart = self._indexNearest(edgeXList, cellList[a]["coordinate"][0])
            columnEnd = self._indexNearest(edgeXList, cellList[a]["coordinate"][2])
            rowStart = self._indexNearest(edgeYList, cellList[a]["coordinate"][1])
            rowEnd = self._indexNearest(edgeYList, cellList[a]["coordinate"][3])

            cellList[a]["columnIndex"] = columnStart
            cellList[a]["rowIndex"] = rowStart
            cellList[a]["columnSpan"] = max(1, columnEnd - columnStart)
            cellList[a]["rowSpan"] = max(1, rowEnd - rowStart)

        cellList = self._cellMerge(cellList)

        cellList.sort(key=lambda cellObject: (cellObject["rowIndex"], cellObject["columnIndex"]))

        return {
            "rowCount": max(0, len(edgeYList) - 1),
            "columnCount": max(0, len(edgeXList) - 1),
            "edgeXList": edgeXList,
            "edgeYList": edgeYList,
            "cellList": cellList
        }

    def execute(self, coordinateList, image):
        x1 = int(round(coordinateList[0]))
        y1 = int(round(coordinateList[1]))
        x2 = int(round(coordinateList[2]))
        y2 = int(round(coordinateList[3]))

        imageRgb = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)

        tableType = self._typeClassify(imageRgb)

        cellList = self._inference(imageRgb, tableType)

        for a in range(len(cellList)):
            cellList[a]["coordinate"] = [
                cellList[a]["coordinate"][0] + x1,
                cellList[a]["coordinate"][1] + y1,
                cellList[a]["coordinate"][2] + x1,
                cellList[a]["coordinate"][3] + y1
            ]

        gridObject = self._gridBuild(cellList, x2 - x1, y2 - y1)

        gridObject["type"] = tableType

        return gridObject

    def __init__(self):
        self.osPathDirName = f"{os.path.dirname(__file__)}/"
        self.pathModelClassification = f"{self.osPathDirName}model/pp-lcNet_x1_0_table_cls.onnx"
        self.pathModelDetectionObject = {
            "wired": f"{self.osPathDirName}model/rt-detr-l_wired_table_cell_det.onnx",
            "wireless": f"{self.osPathDirName}model/rt-detr-l_wireless_table_cell_det.onnx"
        }

        self.imageSizeShort = 256
        self.imageSizeCrop = 224
        self.imageSizeDetection = 640

        self.normalizeMeanList = numpy.array([0.485, 0.456, 0.406], dtype=numpy.float32)
        self.normalizeStdList = numpy.array([0.229, 0.224, 0.225], dtype=numpy.float32)

        self.levelBoxContained = 0.9
        self.levelBoxNms = 0.5
        self.levelEdgeSnap = 0.02
        self.levelEdgeSupport = 0.3

        self.countContainedMinimum = 2

        self.scoreThreshold = 0.3

        self.labelList = ["wired", "wireless"]

        self.onnxSessionClassification = onnxSessionBuild(self.pathModelClassification)

        self.onnxSessionDetectionObject = {}

        for a in range(len(self.labelList)):
            self.onnxSessionDetectionObject[self.labelList[a]] = onnxSessionBuild(self.pathModelDetectionObject[self.labelList[a]])


class Vector:
    def _segmentSelect(self, segmentList, coordinateList, isHorizontal):
        resultList = []

        for a in range(len(segmentList)):
            segment = segmentList[a]

            if isHorizontal:
                if segment["position"] < coordinateList[1] - self.marginBox or segment["position"] > coordinateList[3] + self.marginBox:
                    continue

                start = max(segment["start"], coordinateList[0])
                end = min(segment["end"], coordinateList[2])
            else:
                if segment["position"] < coordinateList[0] - self.marginBox or segment["position"] > coordinateList[2] + self.marginBox:
                    continue

                start = max(segment["start"], coordinateList[1])
                end = min(segment["end"], coordinateList[3])

            if end - start < self.lengthMinimum:
                continue

            resultList.append({"position": segment["position"], "start": start, "end": end})

        return resultList

    def _intervalMerge(self, intervalList):
        resultList = []

        intervalSortedList = sorted(intervalList, key=lambda intervalObject: intervalObject[0])

        for a in range(len(intervalSortedList)):
            if len(resultList) > 0 and intervalSortedList[a][0] <= resultList[len(resultList) - 1][1] + self.tolerance:
                resultList[len(resultList) - 1][1] = max(resultList[len(resultList) - 1][1], intervalSortedList[a][1])
            else:
                resultList.append([intervalSortedList[a][0], intervalSortedList[a][1]])

        return resultList

    def _edgeBuild(self, segmentList):
        resultList = []

        segmentSortedList = sorted(segmentList, key=lambda segmentObject: segmentObject["position"])

        for a in range(len(segmentSortedList)):
            if len(resultList) > 0 and segmentSortedList[a]["position"] - resultList[len(resultList) - 1]["position"] <= self.tolerance:
                resultList[len(resultList) - 1]["intervalList"].append([segmentSortedList[a]["start"], segmentSortedList[a]["end"]])
            else:
                resultList.append({
                    "position": segmentSortedList[a]["position"],
                    "intervalList": [[segmentSortedList[a]["start"], segmentSortedList[a]["end"]]]
                })

        for a in range(len(resultList)):
            resultList[a]["intervalList"] = self._intervalMerge(resultList[a]["intervalList"])

        return resultList

    def _coverageCheck(self, intervalList, start, end):
        length = end - start

        if length <= 0:
            return True

        covered = 0.0

        for a in range(len(intervalList)):
            covered += max(0.0, min(intervalList[a][1], end) - max(intervalList[a][0], start))

        return covered / length >= self.levelCoverage

    def _edgeFilter(self, edgeList, start, end, level):
        resultList = []

        length = end - start

        for a in range(len(edgeList)):
            covered = 0.0

            intervalList = edgeList[a]["intervalList"]

            for b in range(len(intervalList)):
                covered += max(0.0, min(intervalList[b][1], end) - max(intervalList[b][0], start))

            if a == 0 or a == len(edgeList) - 1 or covered / length >= level:
                resultList.append(edgeList[a])

        return resultList

    def _positionList(self, edgeList):
        resultList = []

        for a in range(len(edgeList)):
            resultList.append(edgeList[a]["position"])

        return resultList

    def execute(self, coordinateList, segmentObject):
        edgeXList = self._edgeBuild(self._segmentSelect(segmentObject["verticalList"], coordinateList, False))
        edgeYList = self._edgeBuild(self._segmentSelect(segmentObject["horizontalList"], coordinateList, True))

        edgeXList = self._edgeFilter(edgeXList, coordinateList[1], coordinateList[3], self.levelEdgeColumn)
        edgeYList = self._edgeFilter(edgeYList, coordinateList[0], coordinateList[2], self.levelEdgeRow)

        if len(edgeXList) < 2 or len(edgeYList) < 2:
            return {"rowCount": 0, "columnCount": 0, "edgeXList": [], "edgeYList": [], "cellList": [], "type": "wired"}

        rowCount = len(edgeYList) - 1
        columnCount = len(edgeXList) - 1

        consumedObject = {}

        cellList = []

        for a in range(rowCount):
            for b in range(columnCount):
                if (a, b) in consumedObject:
                    continue

                columnSpan = 1

                while b + columnSpan < columnCount and not self._coverageCheck(
                    edgeXList[b + columnSpan]["intervalList"], edgeYList[a]["position"], edgeYList[a + 1]["position"]
                ):
                    columnSpan += 1

                rowSpan = 1

                while a + rowSpan < rowCount and not self._coverageCheck(
                    edgeYList[a + rowSpan]["intervalList"], edgeXList[b]["position"], edgeXList[b + columnSpan]["position"]
                ):
                    rowSpan += 1

                for c in range(rowSpan):
                    for d in range(columnSpan):
                        consumedObject[(a + c, b + d)] = True

                cellList.append({
                    "score": 1.0,
                    "coordinate": [
                        edgeXList[b]["position"],
                        edgeYList[a]["position"],
                        edgeXList[b + columnSpan]["position"],
                        edgeYList[a + rowSpan]["position"]
                    ],
                    "rowIndex": a,
                    "columnIndex": b,
                    "rowSpan": rowSpan,
                    "columnSpan": columnSpan
                })

        return {
            "rowCount": rowCount,
            "columnCount": columnCount,
            "edgeXList": self._positionList(edgeXList),
            "edgeYList": self._positionList(edgeYList),
            "cellList": cellList,
            "type": "wired"
        }

    def __init__(self):
        self.marginBox = 4.0
        self.lengthMinimum = 8.0
        self.tolerance = 3.0
        self.levelCoverage = 0.8
        self.levelEdgeColumn = 0.0
        self.levelEdgeRow = 0.4
