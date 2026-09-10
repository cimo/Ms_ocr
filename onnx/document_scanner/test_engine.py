import sys
import os
import cv2
import json
import shutil

sys.dont_write_bytecode = True

# Source
import test_layout
import test_ocr
import test_table

class Processor:
    def _centerPointCalculate(self, coordinateList):
        return {
            "x": int(round((coordinateList[0] + coordinateList[2]) / 2)),
            "y": int(round((coordinateList[1] + coordinateList[3]) / 2))
        }

    def _outputBuild(self, pathOutput):
        if os.path.isdir(pathOutput):
            shutil.rmtree(pathOutput)

        for a in range(len(self.debugNameList)):
            os.makedirs(f"{pathOutput}debug/{self.debugNameList[a]}/", exist_ok=True)

        os.makedirs(f"{pathOutput}page/", exist_ok=True)

    def _pageBuild(self, pathImage, pathOutput):
        image = cv2.imread(pathImage)

        if image is None:
            return []

        cv2.imwrite(f"{pathOutput}page/{self.numberPageFirst}.jpg", image)

        return [{"number": self.numberPageFirst, "image": image}]

    def _layoutBuild(self, astPage):
        resultList = []

        flowObject = {"main": astPage["itemMainList"], "secondary": astPage["itemSecondaryList"]}

        for flow in flowObject:
            itemList = flowObject[flow]

            for a in range(len(itemList)):
                resultList.append({
                    "page": astPage["number"],
                    "flow": flow,
                    "label": itemList[a]["label"],
                    "score": itemList[a]["score"],
                    "bbox": itemList[a]["coordinate"],
                    "centerPoint": self._centerPointCalculate(itemList[a]["coordinate"])
                })

        return resultList

    def _tableCollect(self, astPage):
        resultList = []

        itemList = astPage["itemMainList"] + astPage["itemSecondaryList"]

        for a in range(len(itemList)):
            if itemList[a]["label"] == self.labelTable:
                resultList.append(itemList[a])

        return resultList

    def execute(self, pathImage):
        fileName = os.path.splitext(os.path.basename(pathImage))[0]

        pathOutput = f"{self.pathOutput}{fileName}/"

        self._outputBuild(pathOutput)

        pageList = self._pageBuild(pathImage, pathOutput)

        layoutList = []
        itemList = []

        for a in range(len(pageList)):
            astPage = self.testLayout.execute(pageList[a]["image"], pageList[a]["number"], pathOutput)

            tableList = self.testTable.execute(self._tableCollect(astPage), pageList[a]["image"])

            itemPageList = self.testOcr.execute(pageList[a]["image"], tableList, pageList[a]["number"], pathOutput)

            self.testTable.debugWrite(tableList, pageList[a]["image"], itemPageList, pathOutput, pageList[a]["number"])

            layoutList = layoutList + self._layoutBuild(astPage)
            itemList = itemList + itemPageList

        with open(f"{pathOutput}{self.resultFileName}", "w", encoding="utf-8") as file:
            json.dump({"layoutList": layoutList, "itemList": itemList}, file, ensure_ascii=False, indent=2)

        with open(f"{pathOutput}{self.markdownFileName}", "w", encoding="utf-8") as file:
            file.write("")

    def __init__(self):
        self.osPathDirName = f"{os.path.dirname(__file__)}/"
        self.pathOutput = f"{self.osPathDirName}../../file/output/"

        self.resultFileName = "result.json"
        self.markdownFileName = "result.md"

        self.numberPageFirst = 1

        self.countThread = 0

        self.labelTable = "table"

        self.debugNameList = ["layout", "ocr", "table"]

        self.testLayout = test_layout.Test()
        self.testOcr = test_ocr.Test()
        self.testTable = test_table.Test()

        cv2.setUseOptimized(True)
        cv2.setNumThreads(self.countThread)

if __name__ == "__main__":
    processor = Processor()

    for a in range(1, len(sys.argv)):
        processor.execute(sys.argv[a])
