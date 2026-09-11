import sys
import cv2

sys.dont_write_bytecode = True

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

            itemPageList = self.ocr.execute(pageList[a]["image"], tablePageList, len(itemList), pageList[a]["number"], pathOutput)

            self.table.cellRefine(tablePageList, itemPageList)

            self.table.textAssign(tablePageList, itemPageList)

            self.table.debugWrite(tablePageList, pageList[a]["image"], itemPageList, pathOutput, pageList[a]["number"], len(tableList))

            tableList = tableList + self.table.resultBuild(tablePageList, len(tableList), pageList[a]["number"])
            itemList = itemList + itemPageList

        self.layout.flowAssign(astPageList)

        for a in range(len(astPageList)):
            layoutList = layoutList + self.layout.resultBuild(astPageList[a], len(layoutList))

        self.layout.astWrite(pathOutput, astPageList)

        return {
            "pageCount": len(pageList),
            "layoutList": layoutList,
            "tableList": tableList,
            "itemList": itemList
        }

    def __init__(self, layout, table, ocr):
        self.numberPageFirst = 1

        self.layout = layout
        self.table = table
        self.ocr = ocr
