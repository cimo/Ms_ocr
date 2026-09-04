import sys
sys.dont_write_bytecode = True

import os
import shutil
import time
import json
import unicodedata
import cv2
import numpy

# Source
from layout import Layout
from detection import Detection
from recognition import Recognition

class Engine:
    def _centerPointCalculate(self, pointList):
        xList = []
        yList = []

        for a in range(len(pointList)):
            xList.append(pointList[a][0])
            yList.append(pointList[a][1])

        return {
            "x": int(round((min(xList) + max(xList)) / 2)),
            "y": int(round((min(yList) + max(yList)) / 2))
        }

    def _matchCheck(self, value, searchText):
        if searchText == "" or value == "":
            return False

        text = unicodedata.normalize("NFKC", value).strip().casefold().replace(" ", "")
        textSearch = unicodedata.normalize("NFKC", searchText).strip().casefold().replace(" ", "")

        return textSearch in text

    def _debugDrawLayout(self, image, layoutList, fileName, pathDebug):
        imageCopy = image.copy()

        labelDrawnList = []

        for a in range(len(layoutList)):
            coordinateList = layoutList[a]["coordinate"]

            cv2.rectangle(imageCopy, (coordinateList[0], coordinateList[1]), (coordinateList[2], coordinateList[3]), (255, 0, 0), 1)

            text = f"{layoutList[a]['label']} {layoutList[a]['score']:.2f}"

            textWidth, textHeight = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]

            x = coordinateList[0]
            y = coordinateList[1] - 4

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

            cv2.putText(
                imageCopy,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
                cv2.LINE_AA
            )

        cv2.imwrite(f"{pathDebug}{fileName}", imageCopy)

    def _debugDrawItem(self, image, detectionList, resultItemList, fileName, pathDebug):
        imageCopy = image.copy()

        for a in range(len(detectionList)):
            color = (0, 200, 0) if resultItemList[a]["isMatch"] else (0, 0, 255)

            cv2.polylines(imageCopy, [numpy.array(detectionList[a]["coordinate"], dtype=numpy.int32)], True, color, 1)

        cv2.imwrite(f"{pathDebug}{fileName}", imageCopy)

    def execute(self, pathInput, pathOutput, fileName, searchText):
        timeStart = time.perf_counter()

        pathDebug = f"{pathOutput}debug/"

        if os.path.isdir(pathDebug):
            shutil.rmtree(pathDebug)

        if self.isDebug:
            os.makedirs(pathDebug, exist_ok=True)

        image = cv2.imread(pathInput)

        if image is None:
            return {"layoutList": [], "itemList": []}

        imageRgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        layoutList = self.layout.execute(imageRgb)
        detectionList = self.detection.execute(image)

        resultLayoutList = []

        for a in range(len(layoutList)):
            coordinateList = layoutList[a]["coordinate"]

            resultLayoutList.append({
                "label": layoutList[a]["label"],
                "score": layoutList[a]["score"],
                "centerPoint": self._centerPointCalculate([[coordinateList[0], coordinateList[1]], [coordinateList[2], coordinateList[3]]])
            })

        resultItemList = []

        for a in range(len(detectionList)):
            recognitionObject = self.recognition.execute(image, detectionList[a]["coordinate"])

            resultItemList.append({
                "id": a + 1,
                "centerPoint": self._centerPointCalculate(detectionList[a]["coordinate"]),
                "text": recognitionObject["text"],
                "isMatch": self._matchCheck(recognitionObject["text"], searchText)
            })

        os.makedirs(pathOutput, exist_ok=True)

        with open(f"{pathOutput}result.json", "w", encoding="utf-8") as file:
            json.dump({"layoutList": resultLayoutList, "itemList": resultItemList}, file, ensure_ascii=False, indent=2)

        if self.isDebug:
            self._debugDrawLayout(image, layoutList, "layout.jpg", pathDebug)
            self._debugDrawItem(image, detectionList, resultItemList, "item.jpg", pathDebug)

        timeEnd = time.perf_counter() - timeStart

        print(f"\nEngine.py - Time: {round(timeEnd, 3)} - {fileName}")

        return {"layoutList": resultLayoutList, "itemList": resultItemList}

    def __init__(self):
        self.isDebug = os.environ["MS_O_IS_DEBUG"] == "true"

        self.layout = Layout()
        self.detection = Detection()
        self.recognition = Recognition()
