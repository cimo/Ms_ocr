import sys
import os
import cv2
import json
import numpy

sys.dont_write_bytecode = True
sys.path.append(f"{os.path.dirname(__file__)}/..")

# Source
import detection
import recognition

class Test:
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

    def _debugDraw(self, image, coordinateItemList, fileName):
        imageDebug = image.copy()

        for a in range(len(coordinateItemList)):
            coordinateList = coordinateItemList[a]

            boxRegion = imageDebug[coordinateList[1]:coordinateList[3], coordinateList[0]:coordinateList[2]]
            boxOverlay = numpy.full(boxRegion.shape, self.colorText, dtype=numpy.uint8)

            imageDebug[coordinateList[1]:coordinateList[3], coordinateList[0]:coordinateList[2]] = cv2.addWeighted(boxOverlay, self.levelDebugOpacity, boxRegion, 1 - self.levelDebugOpacity, 0)

        cv2.imwrite(f"{self.pathOutput}{fileName}/{self.debugFileName}", imageDebug)

    def execute(self, pathImage):
        fileName = os.path.splitext(os.path.basename(pathImage))[0]

        image = cv2.imread(pathImage)

        if image is None:
            return

        detectionList = self.detection.execute(image)

        itemList = []
        coordinateItemList = []

        for a in range(len(detectionList)):
            recognitionObject = self.recognition.execute(detectionList[a]["coordinate"], image)

            if len(recognitionObject["text"].strip()) == 0:
                continue

            coordinateList = self._coordinateCalculate(detectionList[a]["coordinate"])

            itemList.append({
                "id": len(itemList) + 1,
                "page": self.numberPage,
                "centerPoint": self._centerPointCalculate(coordinateList),
                "text": recognitionObject["text"],
                "isMatch": False
            })

            coordinateItemList.append(coordinateList)

        os.makedirs(f"{self.pathOutput}{fileName}/", exist_ok=True)

        self._debugDraw(image, coordinateItemList, fileName)

        with open(f"{self.pathOutput}{fileName}/{self.resultFileName}", "w", encoding="utf-8") as file:
            json.dump({"layoutList": [], "itemList": itemList}, file, ensure_ascii=False, indent=2)

    def __init__(self):
        self.osPathDirName = f"{os.path.dirname(__file__)}/"
        self.pathOutput = f"{self.osPathDirName}../../file/output/ocr/"

        self.resultFileName = "result.json"
        self.debugFileName = "debug.jpg"

        self.levelDebugOpacity = 0.2

        self.colorText = (0, 255, 255)

        self.numberPage = 1

        cv2.setUseOptimized(True)
        cv2.setNumThreads(1)

        self.detection = detection.Detection()
        self.recognition = recognition.Recognition()

if __name__ == "__main__":
    test = Test()

    for a in range(1, len(sys.argv)):
        test.execute(sys.argv[a])
