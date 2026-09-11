import sys
import os
import cv2
import numpy

sys.dont_write_bytecode = True
sys.path.append(f"{os.path.dirname(__file__)}/..")

# Source
from helper import onnxSessionBuild

class Table:
    def _collect(self, astPage):
        resultList = []

        itemList = astPage["itemMainList"] + astPage["itemSecondaryList"]

        for a in range(len(itemList)):
            if itemList[a]["label"] == "table":
                resultList.append(itemList[a])

        return resultList

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

        scoreList = tensorOutputList[0][0]

        index = int(numpy.argmax(scoreList))

        return {
            "type": self.labelList[index],
            "score": float(scoreList[index])
        }

    def _cellDetect(self, imageRgb, tableType):
        imageHeight, imageWidth = imageRgb.shape[0:2]

        imageResized = cv2.resize(imageRgb, (self.imageSizeCell, self.imageSizeCell), interpolation=cv2.INTER_CUBIC).astype(numpy.float32) / 255.0

        tensor = numpy.expand_dims(imageResized.transpose((2, 0, 1)), axis=0).astype(numpy.float32)

        tensorFeedObject = {
            "image": tensor,
            "im_shape": numpy.array([[self.imageSizeCell, self.imageSizeCell]], dtype=numpy.float32),
            "scale_factor": numpy.array([[self.imageSizeCell / float(imageHeight), self.imageSizeCell / float(imageWidth)]], dtype=numpy.float32)
        }

        onnxSessionCell = self.onnxSessionCellWired if tableType == "wired" else self.onnxSessionCellWireless

        tensorOutputList = onnxSessionCell.run(None, tensorFeedObject)

        boxCount = int(tensorOutputList[1][0])

        scoreThreshold = self.scoreThresholdCellWired if tableType == "wired" else self.scoreThresholdCellWireless

        resultList = []

        for a in range(boxCount):
            value = tensorOutputList[0][a]

            score = float(value[1])

            if score < scoreThreshold:
                continue

            x1 = max(0, min(int(round(float(value[2]))), imageWidth))
            y1 = max(0, min(int(round(float(value[3]))), imageHeight))
            x2 = max(0, min(int(round(float(value[4]))), imageWidth))
            y2 = max(0, min(int(round(float(value[5]))), imageHeight))

            if x2 <= x1 or y2 <= y1:
                continue

            resultList.append({
                "score": score,
                "coordinate": [x1, y1, x2, y2]
            })

        cellList = self._boxSuppression(resultList)
        cellList = self._cellContainedRemove(cellList)
        cellList = self._boxOverlapRemove(cellList)

        return cellList

    def _boxSuppression(self, boxList):
        resultList = []

        boxSortedList = sorted(boxList, key=lambda boxObject: boxObject["score"], reverse=True)

        for a in range(len(boxSortedList)):
            coordinateList = boxSortedList[a]["coordinate"]

            area = (coordinateList[2] - coordinateList[0]) * (coordinateList[3] - coordinateList[1])

            isOverlapped = False

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

                if areaIntersection / float(area + areaKept - areaIntersection) >= self.levelBoxNms:
                    isOverlapped = True

                    break

            if isOverlapped == False:
                resultList.append(boxSortedList[a])

        return resultList

    def _cellContainedRemove(self, cellList):
        resultList = []

        for a in range(len(cellList)):
            coordinateList = cellList[a]["coordinate"]

            countContained = 0

            for b in range(len(cellList)):
                if a == b:
                    continue

                coordinateChildList = cellList[b]["coordinate"]

                areaChild = (coordinateChildList[2] - coordinateChildList[0]) * (coordinateChildList[3] - coordinateChildList[1])

                x1 = max(coordinateList[0], coordinateChildList[0])
                y1 = max(coordinateList[1], coordinateChildList[1])
                x2 = min(coordinateList[2], coordinateChildList[2])
                y2 = min(coordinateList[3], coordinateChildList[3])

                if x2 <= x1 or y2 <= y1:
                    continue

                if (x2 - x1) * (y2 - y1) / float(areaChild) >= self.levelBoxContained:
                    countContained += 1

            if countContained < self.countContainedMinimum:
                resultList.append(cellList[a])

        return self._boxContainedRemove(resultList)

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

    def _boxOverlapRemove(self, boxList):
        resultList = []

        for a in range(len(boxList)):
            coordinateList = boxList[a]["coordinate"]

            area = (coordinateList[2] - coordinateList[0]) * (coordinateList[3] - coordinateList[1])

            areaOverlap = 0

            for b in range(len(boxList)):
                if a == b or boxList[b]["score"] <= boxList[a]["score"]:
                    continue

                coordinateOtherList = boxList[b]["coordinate"]

                x1 = max(coordinateList[0], coordinateOtherList[0])
                y1 = max(coordinateList[1], coordinateOtherList[1])
                x2 = min(coordinateList[2], coordinateOtherList[2])
                y2 = min(coordinateList[3], coordinateOtherList[3])

                if x2 <= x1 or y2 <= y1:
                    continue

                areaOverlap += (x2 - x1) * (y2 - y1)

            if areaOverlap / float(area) < self.levelBoxOverlap:
                resultList.append(boxList[a])

        return resultList

    def _cellRecover(self, imageRgb, cellList):
        coverageList = self._coverageCollect(imageRgb, cellList)

        resultList = []

        for a in range(len(cellList)):
            resultList.append(cellList[a])

        for a in range(len(coverageList)):
            resultList.append({"score": self.scoreCellRecovered, "coordinate": coverageList[a]})

        return resultList

    def _coverageCollect(self, imageRgb, cellList):
        if len(cellList) == 0:
            return []

        imageHeight, imageWidth = imageRgb.shape[0:2]

        imageMask = numpy.ones((imageHeight, imageWidth), dtype=numpy.uint8)

        heightList = []

        for a in range(len(cellList)):
            coordinateList = cellList[a]["coordinate"]

            imageMask[coordinateList[1]:coordinateList[3], coordinateList[0]:coordinateList[2]] = 0

            heightList.append(coordinateList[3] - coordinateList[1])

        heightList.sort()

        sizeKernel = max(self.sizeCoverageKernel, int(round(heightList[int(len(heightList) / 2)] * self.levelCoverageKernel)))

        imageMask = cv2.morphologyEx(imageMask, cv2.MORPH_OPEN, numpy.ones((sizeKernel, sizeKernel), dtype=numpy.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)

        contourList = cv2.findContours(imageMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        resultList = []

        for a in range(len(contourList)):
            if cv2.contourArea(contourList[a]) < imageWidth * imageHeight * self.levelCoverageArea:
                continue

            x, y, width, height = cv2.boundingRect(contourList[a])

            resultList.append([x, y, x + width, y + height])

        return self._coverageFilter(self._coverageSplit(resultList, cellList), cellList)

    def _coverageSplit(self, coverageList, cellList):
        resultList = []

        for a in range(len(coverageList)):
            coverageCoordinateList = coverageList[a]

            margin = max(self.marginCoverage, (coverageCoordinateList[2] - coverageCoordinateList[0]) * self.levelMarginCoverage)

            edgeList = []

            for b in range(len(cellList)):
                coordinateList = cellList[b]["coordinate"]

                for c in range(2):
                    edge = coordinateList[c * 2]

                    if edge > coverageCoordinateList[0] + margin and edge < coverageCoordinateList[2] - margin:
                        edgeList.append(edge)

            edgeList.sort()

            positionList = [coverageCoordinateList[0]]

            for b in range(len(edgeList)):
                if edgeList[b] - positionList[len(positionList) - 1] > margin:
                    positionList.append(edgeList[b])

            positionList.append(coverageCoordinateList[2])

            for b in range(len(positionList) - 1):
                resultList.append([positionList[b], coverageCoordinateList[1], positionList[b + 1], coverageCoordinateList[3]])

        return resultList

    def _coverageFilter(self, coverageList, cellList):
        widthList = []
        heightList = []

        for a in range(len(cellList)):
            coordinateList = cellList[a]["coordinate"]

            widthList.append(coordinateList[2] - coordinateList[0])
            heightList.append(coordinateList[3] - coordinateList[1])

        widthList.sort()
        heightList.sort()

        widthMinimum = widthList[int(len(widthList) / 2)] * self.levelCoverageSize
        heightMinimum = heightList[int(len(heightList) / 2)] * self.levelCoverageSize

        resultList = []

        for a in range(len(coverageList)):
            if coverageList[a][2] - coverageList[a][0] < widthMinimum or coverageList[a][3] - coverageList[a][1] < heightMinimum:
                continue

            resultList.append(coverageList[a])

        return resultList

    def _gridBuild(self, cellList):
        if len(cellList) == 0:
            return cellList

        widthList = []
        heightList = []

        xList = []
        yList = []

        for a in range(len(cellList)):
            coordinateList = cellList[a]["coordinate"]

            widthList.append(coordinateList[2] - coordinateList[0])
            heightList.append(coordinateList[3] - coordinateList[1])

            xList.append(coordinateList[0])
            xList.append(coordinateList[2])

            yList.append(coordinateList[1])
            yList.append(coordinateList[3])

        widthList.sort()
        heightList.sort()

        toleranceColumn = widthList[int(len(widthList) / 2)] * self.levelGridTolerance
        toleranceRow = heightList[int(len(heightList) / 2)] * self.levelGridTolerance

        columnPositionList = self._positionFilter(self._positionCluster(xList, toleranceColumn), xList, toleranceColumn)
        rowPositionList = self._positionCluster(yList, toleranceRow)

        for a in range(len(cellList)):
            coordinateList = cellList[a]["coordinate"]

            columnIndex = self._positionIndex(coordinateList[0], columnPositionList)
            rowIndex = self._positionIndex(coordinateList[1], rowPositionList)

            cellList[a]["rowIndex"] = rowIndex
            cellList[a]["columnIndex"] = columnIndex
            cellList[a]["rowSpan"] = max(1, self._positionIndex(coordinateList[3], rowPositionList) - rowIndex)
            cellList[a]["columnSpan"] = max(1, self._positionIndex(coordinateList[2], columnPositionList) - columnIndex)

        return sorted(cellList, key=lambda cellObject: (cellObject["rowIndex"], cellObject["columnIndex"]))

    def _positionCluster(self, valueList, tolerance):
        resultList = []

        valueSortedList = sorted(valueList)

        for a in range(len(valueSortedList)):
            if len(resultList) > 0 and valueSortedList[a] - resultList[len(resultList) - 1] <= tolerance:
                continue

            resultList.append(valueSortedList[a])

        return resultList

    def _positionFilter(self, positionList, valueList, tolerance):
        countList = []

        for a in range(len(positionList)):
            count = 0

            for b in range(len(valueList)):
                if abs(valueList[b] - positionList[a]) <= tolerance:
                    count += 1

            countList.append(count)

        countMinimum = max(countList) * self.levelGridSupport

        resultList = []

        for a in range(len(positionList)):
            if countList[a] < countMinimum:
                continue

            resultList.append(positionList[a])

        return resultList

    def _positionIndex(self, value, positionList):
        indexNearest = 0
        distanceNearest = abs(value - positionList[0])

        for a in range(1, len(positionList)):
            distance = abs(value - positionList[a])

            if distance >= distanceNearest:
                continue

            indexNearest = a
            distanceNearest = distance

        return indexNearest

    def _textCollect(self, itemList, coordinateTableList):
        resultList = []

        for a in range(len(itemList)):
            coordinateList = [
                itemList[a]["bbox"][0] - coordinateTableList[0],
                itemList[a]["bbox"][1] - coordinateTableList[1],
                itemList[a]["bbox"][2] - coordinateTableList[0],
                itemList[a]["bbox"][3] - coordinateTableList[1]
            ]

            centerX = (coordinateList[0] + coordinateList[2]) / 2
            centerY = (coordinateList[1] + coordinateList[3]) / 2

            if centerX < 0 or centerY < 0 or centerX > coordinateTableList[2] - coordinateTableList[0] or centerY > coordinateTableList[3] - coordinateTableList[1]:
                continue

            resultList.append({"coordinate": coordinateList, "text": itemList[a]["text"]})

        return resultList

    def _cellSplit(self, cellList, textList):
        resultList = []

        for a in range(len(cellList)):
            coordinateList = cellList[a]["coordinate"]

            lineList = self._lineGroup(self._textInsideCollect(textList, coordinateList))

            if len(lineList) < 2:
                resultList.append(cellList[a])

                continue

            positionList = [coordinateList[1]]

            for b in range(len(lineList) - 1):
                positionList.append(int(round((lineList[b]["y2"] + lineList[b + 1]["y1"]) / 2)))

            positionList.append(coordinateList[3])

            for b in range(len(positionList) - 1):
                resultList.append({"score": cellList[a]["score"], "coordinate": [coordinateList[0], positionList[b], coordinateList[2], positionList[b + 1]]})

        return resultList

    def _textInsideCollect(self, textList, cellCoordinateList):
        resultList = []

        for a in range(len(textList)):
            coordinateList = textList[a]["coordinate"]

            centerX = (coordinateList[0] + coordinateList[2]) / 2
            centerY = (coordinateList[1] + coordinateList[3]) / 2

            if centerX < cellCoordinateList[0] or centerX > cellCoordinateList[2] or centerY < cellCoordinateList[1] or centerY > cellCoordinateList[3]:
                continue

            resultList.append(textList[a])

        return resultList

    def _lineGroup(self, textList):
        resultList = []

        textSortedList = sorted(textList, key=lambda textObject: textObject["coordinate"][1])

        for a in range(len(textSortedList)):
            coordinateList = textSortedList[a]["coordinate"]

            isAdded = False

            for b in range(len(resultList)):
                y1 = max(coordinateList[1], resultList[b]["y1"])
                y2 = min(coordinateList[3], resultList[b]["y2"])

                if y2 <= y1:
                    continue

                if (y2 - y1) / float(min(coordinateList[3] - coordinateList[1], resultList[b]["y2"] - resultList[b]["y1"])) < self.levelOverlapLine:
                    continue

                resultList[b]["y1"] = min(resultList[b]["y1"], coordinateList[1])
                resultList[b]["y2"] = max(resultList[b]["y2"], coordinateList[3])

                isAdded = True

                break

            if isAdded == False:
                resultList.append({"y1": coordinateList[1], "y2": coordinateList[3]})

        return sorted(resultList, key=lambda lineObject: lineObject["y1"])

    def _textJoin(self, textList, cellCoordinateList):
        textSortedList = sorted(self._textInsideCollect(textList, cellCoordinateList), key=lambda textObject: (textObject["coordinate"][1], textObject["coordinate"][0]))

        resultList = []

        for a in range(len(textSortedList)):
            resultList.append(textSortedList[a]["text"])

        return " ".join(resultList)

    def _coverageValidate(self, coverageList, textList):
        resultList = []

        for a in range(len(coverageList)):
            coverageCoordinateList = coverageList[a]

            isText = False

            for b in range(len(textList)):
                textCoordinateList = textList[b]["coordinate"]

                centerX = (textCoordinateList[0] + textCoordinateList[2]) / 2
                centerY = (textCoordinateList[1] + textCoordinateList[3]) / 2

                if centerX >= coverageCoordinateList[0] and centerX <= coverageCoordinateList[2] and centerY >= coverageCoordinateList[1] and centerY <= coverageCoordinateList[3]:
                    isText = True

                    break

            resultList.append({"coordinate": coverageCoordinateList, "isText": isText})

        return resultList

    def _textCutCollect(self, textList, cellList):
        resultList = []

        for a in range(len(textList)):
            coordinateList = textList[a]["coordinate"]

            margin = (coordinateList[2] - coordinateList[0]) * self.levelMarginText

            isCut = False

            for b in range(len(cellList)):
                cellCoordinateList = cellList[b]["coordinate"]

                overlap = min(cellCoordinateList[3], coordinateList[3]) - max(cellCoordinateList[1], coordinateList[1])

                if overlap < (coordinateList[3] - coordinateList[1]) * self.levelOverlapText:
                    continue

                for c in range(2):
                    edge = cellCoordinateList[c * 2]

                    if edge > coordinateList[0] + margin and edge < coordinateList[2] - margin:
                        isCut = True

                        break

                if isCut:
                    break

            if isCut:
                resultList.append(coordinateList)

        return resultList

    def _debugCell(self, image, coordinateList, cellList, coverageList, textCutList, pathOutput, numberPage, tableIndex, tableType):
        imageDebug = image[coordinateList[1]:coordinateList[3], coordinateList[0]:coordinateList[2]].copy()

        for a in range(len(cellList)):
            cellCoordinateList = cellList[a]["coordinate"]

            cv2.rectangle(imageDebug, (cellCoordinateList[0], cellCoordinateList[1]), (cellCoordinateList[2], cellCoordinateList[3]), self.colorCell, 1)

        for a in range(len(coverageList)):
            coverageCoordinateList = coverageList[a]["coordinate"]

            color = self.colorCoverage if coverageList[a]["isText"] else self.colorCoverageEmpty

            cv2.rectangle(imageDebug, (coverageCoordinateList[0], coverageCoordinateList[1]), (coverageCoordinateList[2], coverageCoordinateList[3]), color, 1)

        for a in range(len(textCutList)):
            cv2.rectangle(imageDebug, (textCutList[a][0], textCutList[a][1]), (textCutList[a][2], textCutList[a][3]), self.colorTextCut, 1)

        cv2.imwrite(f"{pathOutput}debug/table/{numberPage}_table{tableIndex}_{tableType}.jpg", imageDebug)

    def _centerPointCalculate(self, bboxList):
        return {
            "x": int(round((bboxList[0] + bboxList[2]) / 2)),
            "y": int(round((bboxList[1] + bboxList[3]) / 2))
        }

    def cellRefine(self, tableList, itemList):
        for a in range(len(tableList)):
            textList = self._textCollect(itemList, tableList[a]["coordinate"])

            tableList[a]["cellList"] = self._gridBuild(self._cellSplit(tableList[a]["cellList"], textList))

    def textAssign(self, tableList, itemList):
        for a in range(len(tableList)):
            textList = self._textCollect(itemList, tableList[a]["coordinate"])

            cellList = tableList[a]["cellList"]

            for b in range(len(cellList)):
                cellList[b]["text"] = self._textJoin(textList, cellList[b]["coordinate"])

    def debugWrite(self, tableList, image, itemList, pathOutput, numberPage, countStart):
        for a in range(len(tableList)):
            coordinateList = tableList[a]["coordinate"]

            imageRgb = cv2.cvtColor(image[coordinateList[1]:coordinateList[3], coordinateList[0]:coordinateList[2]], cv2.COLOR_BGR2RGB)

            cellList = tableList[a]["cellList"]

            textList = self._textCollect(itemList, coordinateList)

            coverageList = self._coverageValidate(self._coverageCollect(imageRgb, cellList), textList)

            textCutList = self._textCutCollect(textList, cellList)

            self._debugCell(image, coordinateList, cellList, coverageList, textCutList, pathOutput, numberPage, countStart + a + 1, tableList[a]["type"])

    def resultBuild(self, tablePageList, countStart, numberPage):
        resultList = []

        for a in range(len(tablePageList)):
            coordinateList = tablePageList[a]["coordinate"]

            cellList = tablePageList[a]["cellList"]

            cellResultList = []

            for b in range(len(cellList)):
                cellCoordinateList = cellList[b]["coordinate"]

                bboxList = [
                    cellCoordinateList[0] + coordinateList[0],
                    cellCoordinateList[1] + coordinateList[1],
                    cellCoordinateList[2] + coordinateList[0],
                    cellCoordinateList[3] + coordinateList[1]
                ]

                cellResultList.append({
                    "rowIndex": cellList[b]["rowIndex"],
                    "columnIndex": cellList[b]["columnIndex"],
                    "rowSpan": cellList[b]["rowSpan"],
                    "columnSpan": cellList[b]["columnSpan"],
                    "bbox": bboxList,
                    "centerPoint": self._centerPointCalculate(bboxList),
                    "text": cellList[b]["text"]
                })

            resultList.append({
                "id": countStart + len(resultList) + 1,
                "page": numberPage,
                "type": tablePageList[a]["type"],
                "bbox": coordinateList,
                "centerPoint": self._centerPointCalculate(coordinateList),
                "cellList": cellResultList
            })

        return resultList

    def execute(self, astPage, image):
        resultList = []

        tableList = self._collect(astPage)

        for a in range(len(tableList)):
            coordinateList = tableList[a]["bbox"]

            imageRgb = cv2.cvtColor(image[coordinateList[1]:coordinateList[3], coordinateList[0]:coordinateList[2]], cv2.COLOR_BGR2RGB)

            typeObject = self._typeClassify(imageRgb)

            cellList = self._cellDetect(imageRgb, typeObject["type"])
            cellList = self._cellRecover(imageRgb, cellList)
            cellList = self._gridBuild(cellList)

            resultList.append({
                "coordinate": coordinateList,
                "type": typeObject["type"],
                "cellList": cellList
            })

        return resultList

    def __init__(self):
        self.osPathDirName = f"{os.path.dirname(__file__)}/"
        self.pathModelClassification = f"{self.osPathDirName}model/pp-lcNet_x1_0_table_cls.onnx"
        self.pathModelCellWired = f"{self.osPathDirName}model/rt-detr-l_wired_table_cell_det.onnx"
        self.pathModelCellWireless = f"{self.osPathDirName}model/rt-detr-l_wireless_table_cell_det.onnx"

        self.countContainedMinimum = 2
        self.sizeCoverageKernel = 3
        self.marginCoverage = 4

        self.imageSizeCell = 640
        self.imageSizeShort = 256
        self.imageSizeCrop = 224

        self.normalizeMeanList = numpy.array([0.485, 0.456, 0.406], dtype=numpy.float32)
        self.normalizeStdList = numpy.array([0.229, 0.224, 0.225], dtype=numpy.float32)

        self.levelBoxContained = 0.9
        self.levelBoxNms = 0.5
        self.levelBoxOverlap = 0.7
        self.levelCoverageArea = 0.005
        self.levelCoverageKernel = 0.4
        self.levelCoverageSize = 0.4
        self.levelMarginCoverage = 0.02
        self.levelMarginText = 0.15
        self.levelOverlapLine = 0.5
        self.levelOverlapText = 0.5
        self.levelGridSupport = 0.25
        self.levelGridTolerance = 0.3

        self.scoreThresholdCellWired = 0.3
        self.scoreThresholdCellWireless = 0.15
        self.scoreCellRecovered = 0.0

        self.colorCell = (0, 200, 0)
        self.colorCoverage = (0, 0, 255)
        self.colorCoverageEmpty = (255, 0, 0)
        self.colorTextCut = (255, 0, 255)

        self.labelList = ["wired", "wireless"]

        self.onnxSessionClassification = onnxSessionBuild(self.pathModelClassification)
        self.onnxSessionCellWired = onnxSessionBuild(self.pathModelCellWired)
        self.onnxSessionCellWireless = onnxSessionBuild(self.pathModelCellWireless)
