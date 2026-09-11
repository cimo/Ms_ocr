import sys
import cv2

sys.dont_write_bytecode = True

# Source
import test_layout
import test_ocr
import test_table

class Image:
    def _pageBuild(self, pathInput, pathOutput):
        image = cv2.imread(pathInput)

        if image is None:
            return []

        cv2.imwrite(f"{pathOutput}page/{self.numberPageFirst}.jpg", image)

        return [{"number": self.numberPageFirst, "image": image}]

    def execute(self, pathInput, pathOutput):
        pageList = self._pageBuild(pathInput, pathOutput)

        astPageList = []
        layoutList = []
        tableList = []
        itemList = []

        for a in range(len(pageList)):
            astPage = self.layout.execute(pathOutput, pageList[a]["image"], pageList[a]["number"])

            astPageList.append(astPage)

            tablePageList = self.table.execute(astPage, pageList[a]["image"])

            itemPageList = self.ocr.execute(pageList[a]["image"], tablePageList, pageList[a]["number"], pathOutput)

            self.table.cellRefine(tablePageList, itemPageList)

            self.table.textAssign(tablePageList, itemPageList)

            self.table.debugWrite(tablePageList, pageList[a]["image"], itemPageList, pathOutput, pageList[a]["number"])

            layoutList = layoutList + self.layout.resultBuild(astPage)
            tableList = tableList + self.table.resultBuild(tablePageList, pageList[a]["number"])
            itemList = itemList + itemPageList

        self.layout.astWrite(pathOutput, astPageList)

        return {
            "pageCount": len(pageList),
            "layoutList": layoutList,
            "tableList": tableList,
            "itemList": itemList
        }

    def __init__(self):
        self.numberPageFirst = 1

        self.layout = test_layout.Layout()
        self.ocr = test_ocr.Ocr()
        self.table = test_table.Table()
