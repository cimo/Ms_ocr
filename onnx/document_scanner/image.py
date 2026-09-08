import sys
sys.dont_write_bytecode = True

import os
import cv2
import glob

# Source
import detection
import recognition

class Reader:
    def _coordinateCalculate(self, pointList):
        xList = []
        yList = []

        for a in range(len(pointList)):
            xList.append(pointList[a][0])
            yList.append(pointList[a][1])

        return [min(xList), min(yList), max(xList), max(yList)]

    def _tableCollect(self, astPageList):
        resultObject = {}

        for a in range(len(astPageList)):
            itemList = astPageList[a]["itemMainList"] + astPageList[a]["itemSecondaryList"]

            tableList = []

            for b in range(len(itemList)):
                if itemList[b]["label"] == "table":
                    tableList.append(itemList[b]["tableObject"])

            resultObject[astPageList[a]["number"]] = tableList

        return resultObject

    def _edgeInsideCollect(self, tableList, pointList):
        coordinateList = self._coordinateCalculate(pointList)

        centerY = (coordinateList[1] + coordinateList[3]) / 2

        margin = (coordinateList[3] - coordinateList[1]) * self.levelSplitMargin

        edgeList = []

        for a in range(len(tableList)):
            cellList = tableList[a]["cellList"]

            for b in range(len(cellList)):
                if centerY < cellList[b]["coordinate"][1] or centerY > cellList[b]["coordinate"][3]:
                    continue

                edgeCellList = [cellList[b]["coordinate"][0], cellList[b]["coordinate"][2]]

                for c in range(len(edgeCellList)):
                    if edgeCellList[c] > coordinateList[0] + margin and edgeCellList[c] < coordinateList[2] - margin:
                        edgeList.append(edgeCellList[c])

        edgeList.sort()

        resultList = []

        for a in range(len(edgeList)):
            if len(resultList) == 0 or edgeList[a] - resultList[len(resultList) - 1] > margin:
                resultList.append(edgeList[a])

        return resultList

    def _pointInterpolate(self, pointStart, pointEnd, ratio):
        return [
            pointStart[0] + (pointEnd[0] - pointStart[0]) * ratio,
            pointStart[1] + (pointEnd[1] - pointStart[1]) * ratio
        ]

    def _quadSplit(self, pointList, edgeList):
        if len(edgeList) == 0:
            return [pointList]

        xLeft = min(pointList[0][0], pointList[3][0])
        xRight = max(pointList[1][0], pointList[2][0])

        ratioList = [0.0]

        for a in range(len(edgeList)):
            ratioList.append((edgeList[a] - xLeft) / (xRight - xLeft))

        ratioList.append(1.0)

        resultList = []

        for a in range(len(ratioList) - 1):
            resultList.append([
                self._pointInterpolate(pointList[0], pointList[1], ratioList[a]),
                self._pointInterpolate(pointList[0], pointList[1], ratioList[a + 1]),
                self._pointInterpolate(pointList[3], pointList[2], ratioList[a + 1]),
                self._pointInterpolate(pointList[3], pointList[2], ratioList[a])
            ])

        return resultList

    def execute(self, pathInput, astPageList):
        resultList = []

        fileNameList = sorted(glob.glob(f"{pathInput}*.jpg"), key=lambda path: int(os.path.splitext(os.path.basename(path))[0]))

        tableObject = self._tableCollect(astPageList)

        for a in range(len(fileNameList)):
            pageNumber = int(os.path.splitext(os.path.basename(fileNameList[a]))[0])

            image = cv2.imread(fileNameList[a])

            imageHeight, imageWidth = image.shape[0:2]

            detectionList = self.detection.execute(image)

            tableList = tableObject[pageNumber]

            elementList = []

            for b in range(len(detectionList)):
                quadList = self._quadSplit(detectionList[b]["coordinate"], self._edgeInsideCollect(tableList, detectionList[b]["coordinate"]))

                for c in range(len(quadList)):
                    recognitionObject = self.recognition.execute(quadList[c], image)

                    if len(recognitionObject["text"].strip()) > 0:
                        coordinateList = self._coordinateCalculate(quadList[c])

                        elementList.append({
                            "type": "text",
                            "text": recognitionObject["text"],
                            "x0": coordinateList[0],
                            "y0": coordinateList[1],
                            "x1": coordinateList[2],
                            "y1": coordinateList[3],
                            "fontName": "",
                            "fontSize": coordinateList[3] - coordinateList[1],
                            "isBold": False,
                            "color": "#000000"
                        })

            resultList.append({"number": pageNumber, "width": imageWidth, "height": imageHeight, "elementList": elementList})

        return resultList

    def __init__(self):
        self.levelSplitMargin = 0.5

        self.detection = detection.Detection()
        self.recognition = recognition.Recognition()
