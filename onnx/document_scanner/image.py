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

    def execute(self, pathInput):
        resultList = []

        fileNameList = sorted(glob.glob(f"{pathInput}*.jpg"), key=lambda path: int(os.path.splitext(os.path.basename(path))[0]))

        for a in range(len(fileNameList)):
            pageNumber = int(os.path.splitext(os.path.basename(fileNameList[a]))[0])

            image = cv2.imread(fileNameList[a])

            imageHeight, imageWidth = image.shape[0:2]

            detectionList = self.detection.execute(image)

            elementList = []

            for b in range(len(detectionList)):
                recognitionObject = self.recognition.execute(detectionList[b]["coordinate"], image)

                if len(recognitionObject["text"].strip()) > 0:
                    coordinateList = self._coordinateCalculate(detectionList[b]["coordinate"])

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
        self.detection = detection.Detection()
        self.recognition = recognition.Recognition()
