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
import test_markdown

class Processor:
    def _centerPointCalculate(self, bboxList):
        return {
            "x": int(round((bboxList[0] + bboxList[2]) / 2)),
            "y": int(round((bboxList[1] + bboxList[3]) / 2))
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
                    "id": len(resultList) + 1,
                    "page": astPage["number"],
                    "flow": flow,
                    "label": itemList[a]["label"],
                    "score": itemList[a]["score"],
                    "bbox": itemList[a]["bbox"],
                    "centerPoint": itemList[a]["centerPoint"]
                })

        return resultList

    def _tableBuild(self, tablePageList, numberPage):
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
                "id": len(resultList) + 1,
                "page": numberPage,
                "type": tablePageList[a]["type"],
                "bbox": coordinateList,
                "centerPoint": self._centerPointCalculate(coordinateList),
                "cellList": cellResultList
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
        tableList = []
        itemList = []

        for a in range(len(pageList)):
            astPage = self.testLayout.execute(pageList[a]["image"], pageList[a]["number"], pathOutput)

            tablePageList = self.testTable.execute(self._tableCollect(astPage), pageList[a]["image"])

            itemPageList = self.testOcr.execute(pageList[a]["image"], tablePageList, pageList[a]["number"], pathOutput)

            self.testTable.textAssign(tablePageList, itemPageList)

            self.testTable.debugWrite(tablePageList, pageList[a]["image"], itemPageList, pathOutput, pageList[a]["number"])

            layoutList = layoutList + self._layoutBuild(astPage)
            tableList = tableList + self._tableBuild(tablePageList, pageList[a]["number"])
            itemList = itemList + itemPageList

        with open(f"{pathOutput}{self.resultFileName}", "w", encoding="utf-8") as file:
            json.dump({"layoutList": layoutList, "tableList": tableList, "itemList": itemList}, file, ensure_ascii=False, indent=2)

        with open(f"{pathOutput}{self.markdownFileName}", "w", encoding="utf-8") as file:
            file.write(self.testMarkdown.execute(layoutList, tableList, itemList))

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
        self.testMarkdown = test_markdown.Test()

        cv2.setUseOptimized(True)
        cv2.setNumThreads(self.countThread)

if __name__ == "__main__":
    processor = Processor()

    for a in range(1, len(sys.argv)):
        processor.execute(sys.argv[a])
