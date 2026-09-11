import sys
import os
import cv2

sys.dont_write_bytecode = True
sys.path.append(f"{os.path.dirname(__file__)}/..")

# Source
import test_detection
import test_recognition

class Ocr:
    def _coordinateCalculate(self, pointList):
        xList = []
        yList = []

        for a in range(len(pointList)):
            xList.append(pointList[a][0])
            yList.append(pointList[a][1])

        return [min(xList), min(yList), max(xList), max(yList)]

    def _centerPointCalculate(self, coordinateList):
        return {
            "x": int(round((coordinateList[0] + coordinateList[2]) / 2)),
            "y": int(round((coordinateList[1] + coordinateList[3]) / 2))
        }

    def _edgeSplitCheck(self, coordinateList, edge, imageInk):
        y0 = max(0, int(round(coordinateList[1])))
        y1 = min(imageInk.shape[0], int(round(coordinateList[3])))

        height = y1 - y0

        if height <= 0:
            return False

        window = int(round(height * self.levelSplitMargin))

        x0 = max(0, int(round(edge)) - window)
        x1 = min(imageInk.shape[1], int(round(edge)) + window + 1)

        ratioList = imageInk[y0:y1, x0:x1].sum(axis=0) / float(height)

        gapMinimum = height * self.levelSplitGap
        gapCount = 0

        for a in range(len(ratioList)):
            if ratioList[a] >= self.levelSplitLine:
                return True

            if ratioList[a] == 0.0:
                gapCount += 1

                if gapCount >= gapMinimum:
                    return True
            else:
                gapCount = 0

        return False

    def _edgeInsideCollect(self, tableList, pointList, imageInk):
        coordinateList = self._coordinateCalculate(pointList)

        centerY = (coordinateList[1] + coordinateList[3]) / 2

        margin = (coordinateList[3] - coordinateList[1]) * self.levelSplitMargin

        edgeList = []

        for a in range(len(tableList)):
            offsetX = tableList[a]["coordinate"][0]
            offsetY = tableList[a]["coordinate"][1]

            cellList = tableList[a]["cellList"]

            for b in range(len(cellList)):
                if centerY < cellList[b]["coordinate"][1] + offsetY or centerY > cellList[b]["coordinate"][3] + offsetY:
                    continue

                edgeCellList = [cellList[b]["coordinate"][0] + offsetX, cellList[b]["coordinate"][2] + offsetX]

                for c in range(len(edgeCellList)):
                    if edgeCellList[c] > coordinateList[0] + margin and edgeCellList[c] < coordinateList[2] - margin:
                        edgeList.append(edgeCellList[c])

        edgeList.sort()

        resultList = []

        for a in range(len(edgeList)):
            if len(resultList) > 0 and edgeList[a] - resultList[len(resultList) - 1] <= margin:
                continue

            if self._edgeSplitCheck(coordinateList, edgeList[a], imageInk) == False:
                continue

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

    def _debugText(self, image, coordinateItemList, pathOutput, numberPage):
        imageDebug = image.copy()

        for a in range(len(coordinateItemList)):
            coordinateList = coordinateItemList[a]

            cv2.rectangle(imageDebug, (coordinateList[0], coordinateList[1]), (coordinateList[2], coordinateList[3]), (0, 200, 0), 1)

        cv2.imwrite(f"{pathOutput}debug/ocr/{numberPage}.jpg", imageDebug)

    def execute(self, image, tableList, numberPage, pathOutput):
        detectionList = self.detection.execute(image)

        imageInk = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        quadPageList = []

        for a in range(len(detectionList)):
            quadList = self._quadSplit(detectionList[a]["coordinate"], self._edgeInsideCollect(tableList, detectionList[a]["coordinate"], imageInk))

            for b in range(len(quadList)):
                quadPageList.append(quadList[b])

        recognitionList = self.recognition.executeBatch(quadPageList, image)

        itemList = []
        coordinateItemList = []

        for a in range(len(quadPageList)):
            if len(recognitionList[a]["text"].strip()) == 0:
                continue

            coordinateList = self._coordinateCalculate(quadPageList[a])

            itemList.append({
                "id": len(itemList) + 1,
                "page": numberPage,
                "bbox": [int(round(coordinateList[0])), int(round(coordinateList[1])), int(round(coordinateList[2])), int(round(coordinateList[3]))],
                "centerPoint": self._centerPointCalculate(coordinateList),
                "text": recognitionList[a]["text"],
                "isMatch": False
            })

            coordinateItemList.append(itemList[len(itemList) - 1]["bbox"])

        self._debugText(image, coordinateItemList, pathOutput, numberPage)

        return itemList

    def __init__(self):
        self.levelSplitMargin = 0.5
        self.levelSplitLine = 0.8
        self.levelSplitGap = 0.4

        self.detection = test_detection.Detection()
        self.recognition = test_recognition.Recognition()
