import sys
import os
import cv2
import numpy

sys.dont_write_bytecode = True
sys.path.append(f"{os.path.dirname(__file__)}/..")

# Source
from helper import onnxSessionBuild

class Test:
    def _tableDetect(self, image):
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

            if classId != self.classIdTable or score < self.scoreThreshold:
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

        return self._boxContainedRemove(resultList)

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

        tensorOutputList = self.onnxSessionCellObject[tableType].run(None, tensorFeedObject)

        boxCount = int(tensorOutputList[1][0])

        resultList = []

        for a in range(boxCount):
            value = tensorOutputList[0][a]

            score = float(value[1])

            if score < self.scoreThresholdCellObject[tableType]:
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

        return self._coverageSplit(resultList, cellList)

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

    def _debugLayout(self, image, tableList, fileName):
        imageDebug = image.copy()

        for a in range(len(tableList)):
            coordinateList = tableList[a]["coordinate"]

            boxRegion = imageDebug[coordinateList[1]:coordinateList[3], coordinateList[0]:coordinateList[2]]
            boxOverlay = numpy.full(boxRegion.shape, self.colorTable, dtype=numpy.uint8)

            imageDebug[coordinateList[1]:coordinateList[3], coordinateList[0]:coordinateList[2]] = cv2.addWeighted(boxOverlay, self.levelDebugOpacity, boxRegion, 1 - self.levelDebugOpacity, 0)

            cv2.putText(imageDebug, f"table {a} {tableList[a]['score']:.2f}", (coordinateList[0], max(14, coordinateList[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colorTable, 1)

        cv2.imwrite(f"{self.pathOutput}{fileName}_layout.jpg", imageDebug)

    def _debugCell(self, image, coordinateList, cellList, coverageList, typeObject, fileName, tableIndex):
        imageDebug = image[coordinateList[1]:coordinateList[3], coordinateList[0]:coordinateList[2]].copy()

        for a in range(len(cellList)):
            cellCoordinateList = cellList[a]["coordinate"]

            cv2.rectangle(imageDebug, (cellCoordinateList[0], cellCoordinateList[1]), (cellCoordinateList[2], cellCoordinateList[3]), self.colorCell, 1)

        for a in range(len(coverageList)):
            cv2.rectangle(imageDebug, (coverageList[a][0], coverageList[a][1]), (coverageList[a][2], coverageList[a][3]), self.colorCoverage, 1)

        cv2.imwrite(f"{self.pathOutput}{fileName}_table{tableIndex}_{typeObject['type']}.jpg", imageDebug)

    def execute(self, pathImage):
        fileName = os.path.splitext(os.path.basename(pathImage))[0]

        image = cv2.imread(pathImage)

        if image is None:
            return

        tableList = self._tableDetect(image)

        self._debugLayout(image, tableList, fileName)

        for a in range(len(tableList)):
            coordinateList = tableList[a]["coordinate"]

            imageRgb = cv2.cvtColor(image[coordinateList[1]:coordinateList[3], coordinateList[0]:coordinateList[2]], cv2.COLOR_BGR2RGB)

            typeObject = self._typeClassify(imageRgb)

            cellList = self._cellDetect(imageRgb, typeObject["type"])

            coverageList = self._coverageCollect(imageRgb, cellList)

            self._debugCell(image, coordinateList, cellList, coverageList, typeObject, fileName, a)

    def __init__(self):
        self.osPathDirName = f"{os.path.dirname(__file__)}/"
        self.pathModelLayout = f"{self.osPathDirName}model/pp-docLayout_plus-l.onnx"
        self.pathModelClassification = f"{self.osPathDirName}model/pp-lcNet_x1_0_table_cls.onnx"
        self.pathModelCellObject = {
            "wired": f"{self.osPathDirName}model/rt-detr-l_wired_table_cell_det.onnx",
            "wireless": f"{self.osPathDirName}model/rt-detr-l_wireless_table_cell_det.onnx"
        }
        self.pathOutput = f"{self.osPathDirName}../../file/output/table/"

        self.imageSizeLayout = 800
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
        self.levelDebugOpacity = 0.2
        self.levelMarginCoverage = 0.02

        self.scoreThreshold = 0.3
        self.scoreThresholdCellObject = {
            "wired": 0.3,
            "wireless": 0.15
        }

        self.classIdTable = 8

        self.countContainedMinimum = 2

        self.sizeCoverageKernel = 3

        self.marginCoverage = 4

        self.colorTable = (0, 0, 255)
        self.colorCell = (0, 200, 0)
        self.colorCoverage = (0, 0, 255)

        self.labelList = ["wired", "wireless"]

        cv2.setUseOptimized(True)
        cv2.setNumThreads(1)

        os.makedirs(self.pathOutput, exist_ok=True)

        self.onnxSessionLayout = onnxSessionBuild(self.pathModelLayout)
        self.onnxSessionClassification = onnxSessionBuild(self.pathModelClassification)

        self.onnxSessionCellObject = {}

        for a in range(len(self.labelList)):
            self.onnxSessionCellObject[self.labelList[a]] = onnxSessionBuild(self.pathModelCellObject[self.labelList[a]])

if __name__ == "__main__":
    test = Test()

    for a in range(1, len(sys.argv)):
        test.execute(sys.argv[a])
