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
    def _matchCheck(self, value, searchText):
        if searchText == "" or value == "":
            return False

        text = unicodedata.normalize("NFKC", value).strip().casefold().replace(" ", "")
        textSearch = unicodedata.normalize("NFKC", searchText).strip().casefold().replace(" ", "")

        return textSearch in text

    def _debugDrawLayout(self, image, layoutList, fileName):
        imageCopy = image.copy()

        for a in range(len(layoutList)):
            coordinateList = layoutList[a]["coordinate"]

            cv2.rectangle(imageCopy, (coordinateList[0], coordinateList[1]), (coordinateList[2], coordinateList[3]), (255, 0, 0), 1)

            cv2.putText(
                imageCopy,
                f"{layoutList[a]['label']} {layoutList[a]['score']:.2f}",
                (coordinateList[0], max(0, coordinateList[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
                cv2.LINE_AA
            )

        cv2.imwrite(f"{self.pathDebug}{fileName}", imageCopy)

    def _debugDrawItem(self, image, itemList, fileName):
        imageCopy = image.copy()

        for a in range(len(itemList)):
            color = (0, 200, 0) if itemList[a]["isMatch"] else (0, 0, 255)

            cv2.polylines(imageCopy, [numpy.array(itemList[a]["polygon"], dtype=numpy.int32)], True, color, 1)

        cv2.imwrite(f"{self.pathDebug}{fileName}", imageCopy)

    def execute(self, pathInput, pathOutput, fileName, searchText):
        timeStart = time.perf_counter()

        self.pathDebug = f"{pathOutput}debug/"

        if os.path.isdir(self.pathDebug):
            shutil.rmtree(self.pathDebug)

        if self.isDebug:
            os.makedirs(self.pathDebug, exist_ok=True)

        image = cv2.imread(pathInput)

        if image is None:
            return {"layoutList": [], "itemList": []}

        imageRgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        layoutList = self.layout.execute(imageRgb)
        boxList = self.detection.execute(image)

        itemList = []

        for a in range(len(boxList)):
            itemObject = self.recognition.execute(image, boxList[a]["coordinate"])

            itemList.append({
                "id": a + 1,
                "polygon": boxList[a]["coordinate"],
                "text": itemObject["text"],
                "isMatch": self._matchCheck(itemObject["text"], searchText)
            })

        os.makedirs(pathOutput, exist_ok=True)

        with open(f"{pathOutput}result.json", "w", encoding="utf-8") as file:
            json.dump({"layoutList": layoutList, "itemList": itemList}, file, ensure_ascii=False, indent=2)

        if self.isDebug:
            self._debugDrawLayout(image, layoutList, "layout.jpg")
            self._debugDrawItem(image, itemList, "item.jpg")

        timeEnd = time.perf_counter() - timeStart

        print(f"\nEngine.py - Time: {round(timeEnd, 3)} - {fileName}")

        return {"layoutList": layoutList, "itemList": itemList}

    def __init__(self):
        self.pathDebug = ""

        self.isDebug = os.environ["MS_O_IS_DEBUG"] == "true"

        self.layout = Layout()
        self.detection = Detection()
        self.recognition = Recognition()
