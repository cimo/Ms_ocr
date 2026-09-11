import sys

sys.dont_write_bytecode = True

class Test:
    def _lineGroup(self, itemList):
        resultList = []

        itemSortedList = sorted(itemList, key=lambda itemObject: (itemObject["page"], itemObject["centerPoint"]["y"]))

        for a in range(len(itemSortedList)):
            bboxList = itemSortedList[a]["bbox"]

            isAdded = False

            for b in range(len(resultList)):
                if resultList[b]["page"] != itemSortedList[a]["page"]:
                    continue

                y1 = max(bboxList[1], resultList[b]["y1"])
                y2 = min(bboxList[3], resultList[b]["y2"])

                if y2 <= y1:
                    continue

                if (y2 - y1) / float(min(bboxList[3] - bboxList[1], resultList[b]["y2"] - resultList[b]["y1"])) < self.levelLineOverlap:
                    continue

                resultList[b]["y1"] = min(resultList[b]["y1"], bboxList[1])
                resultList[b]["y2"] = max(resultList[b]["y2"], bboxList[3])

                resultList[b]["itemList"].append(itemSortedList[a])

                isAdded = True

                break

            if isAdded == False:
                resultList.append({"page": itemSortedList[a]["page"], "y1": bboxList[1], "y2": bboxList[3], "itemList": [itemSortedList[a]]})

        return sorted(resultList, key=lambda lineObject: (lineObject["page"], lineObject["y1"]))

    def _lineText(self, lineObject):
        itemSortedList = sorted(lineObject["itemList"], key=lambda itemObject: itemObject["bbox"][0])

        textList = []

        for a in range(len(itemSortedList)):
            textList.append(itemSortedList[a]["text"])

        return self.separatorText.join(textList)

    def _itemOrphanCollect(self, itemList, layoutList):
        resultList = []

        for a in range(len(itemList)):
            centerPointObject = itemList[a]["centerPoint"]

            isInside = False

            for b in range(len(layoutList)):
                if layoutList[b]["page"] != itemList[a]["page"]:
                    continue

                bboxList = layoutList[b]["bbox"]

                if centerPointObject["x"] < bboxList[0] or centerPointObject["x"] > bboxList[2]:
                    continue

                if centerPointObject["y"] < bboxList[1] or centerPointObject["y"] > bboxList[3]:
                    continue

                isInside = True

                break

            if isInside == False:
                resultList.append(itemList[a])

        return resultList

    def _textBuild(self, itemList, bboxList):
        itemInsideList = []

        for a in range(len(itemList)):
            centerPointObject = itemList[a]["centerPoint"]

            if centerPointObject["x"] < bboxList[0] or centerPointObject["x"] > bboxList[2]:
                continue

            if centerPointObject["y"] < bboxList[1] or centerPointObject["y"] > bboxList[3]:
                continue

            itemInsideList.append(itemList[a])

        lineList = self._lineGroup(itemInsideList)

        textList = []

        for a in range(len(lineList)):
            textList.append(self._lineText(lineList[a]))

        return self.separatorText.join(textList)

    def _tableFind(self, tableList, layoutObject):
        for a in range(len(tableList)):
            if tableList[a]["page"] != layoutObject["page"]:
                continue

            if tableList[a]["bbox"] != layoutObject["bbox"]:
                continue

            return tableList[a]

        return None

    def _rowRangeBuild(self, cellList):
        resultObject = {}

        for a in range(len(cellList)):
            if cellList[a]["rowSpan"] > 1:
                continue

            rowIndex = cellList[a]["rowIndex"]

            bboxList = cellList[a]["bbox"]

            if rowIndex not in resultObject:
                resultObject[rowIndex] = {"y1": bboxList[1], "y2": bboxList[3]}

                continue

            resultObject[rowIndex]["y1"] = min(resultObject[rowIndex]["y1"], bboxList[1])
            resultObject[rowIndex]["y2"] = max(resultObject[rowIndex]["y2"], bboxList[3])

        return resultObject

    def _rowIndexAnchor(self, cellObject, itemList, rowRangeObject):
        bboxList = cellObject["bbox"]

        y1List = []
        y2List = []

        for a in range(len(itemList)):
            centerPointObject = itemList[a]["centerPoint"]

            if centerPointObject["x"] < bboxList[0] or centerPointObject["x"] > bboxList[2]:
                continue

            if centerPointObject["y"] < bboxList[1] or centerPointObject["y"] > bboxList[3]:
                continue

            y1List.append(itemList[a]["bbox"][1])
            y2List.append(itemList[a]["bbox"][3])

        rowIndexBest = cellObject["rowIndex"]
        overlapBest = 0

        for a in range(cellObject["rowSpan"]):
            rowIndex = cellObject["rowIndex"] + a

            if rowIndex not in rowRangeObject:
                continue

            overlap = min(max(y2List), rowRangeObject[rowIndex]["y2"]) - max(min(y1List), rowRangeObject[rowIndex]["y1"])

            if overlap <= overlapBest:
                continue

            overlapBest = overlap
            rowIndexBest = rowIndex

        return rowIndexBest

    def _tableWrite(self, tableObject, itemList):
        cellList = tableObject["cellList"]

        countRow = 0
        countColumn = 0

        for a in range(len(cellList)):
            if cellList[a]["rowIndex"] + cellList[a]["rowSpan"] > countRow:
                countRow = cellList[a]["rowIndex"] + cellList[a]["rowSpan"]

            if cellList[a]["columnIndex"] + cellList[a]["columnSpan"] > countColumn:
                countColumn = cellList[a]["columnIndex"] + cellList[a]["columnSpan"]

        gridList = []

        for a in range(countRow):
            rowList = []

            for b in range(countColumn):
                rowList.append(self.textCellEmpty)

            gridList.append(rowList)

        rowRangeObject = self._rowRangeBuild(cellList)

        for a in range(len(cellList)):
            text = cellList[a]["text"].replace(self.separatorCell, self.separatorCellEscaped).strip()

            if len(text) == 0:
                continue

            rowIndex = cellList[a]["rowIndex"]

            if cellList[a]["rowSpan"] > 1:
                rowIndex = self._rowIndexAnchor(cellList[a], itemList, rowRangeObject)

            gridList[rowIndex][cellList[a]["columnIndex"]] = text

        headerList = []
        separatorList = []

        for a in range(countColumn):
            headerList.append(self.textCellEmpty)
            separatorList.append(self.textCellSeparator)

        lineList = [
            f"{self.separatorCell} {f' {self.separatorCell} '.join(headerList)} {self.separatorCell}",
            f"{self.separatorCell} {f' {self.separatorCell} '.join(separatorList)} {self.separatorCell}"
        ]

        for a in range(countRow):
            lineList.append(f"{self.separatorCell} {f' {self.separatorCell} '.join(gridList[a])} {self.separatorCell}")

        return self.separatorLine.join(lineList)

    def _blockWrite(self, layoutObject, tableList, itemList):
        label = layoutObject["label"]

        if label in self.labelPlaceholderObject:
            return self.labelPlaceholderObject[label]

        if label == self.labelTable:
            tableObject = self._tableFind(tableList, layoutObject)

            if tableObject != None:
                return self._tableWrite(tableObject, itemList)

        text = self._textBuild(itemList, layoutObject["bbox"])

        if len(text) == 0:
            return ""

        prefix = self.labelPrefixObject[label] if label in self.labelPrefixObject else ""

        return f"{prefix}{text}"

    def execute(self, layoutList, tableList, itemList):
        blockList = []

        lineOrphanList = self._lineGroup(self._itemOrphanCollect(itemList, layoutList))

        indexOrphan = 0

        for a in range(len(layoutList)):
            while indexOrphan < len(lineOrphanList) and (lineOrphanList[indexOrphan]["page"], lineOrphanList[indexOrphan]["y1"]) < (layoutList[a]["page"], layoutList[a]["bbox"][1]):
                blockList.append(self._lineText(lineOrphanList[indexOrphan]))

                indexOrphan += 1

            block = self._blockWrite(layoutList[a], tableList, itemList)

            if len(block) == 0:
                continue

            blockList.append(block)

        while indexOrphan < len(lineOrphanList):
            blockList.append(self._lineText(lineOrphanList[indexOrphan]))

            indexOrphan += 1

        return self.separatorBlock.join(blockList)

    def __init__(self):
        self.levelLineOverlap = 0.5

        self.labelTable = "table"

        self.separatorText = " "
        self.separatorLine = "\n"
        self.separatorBlock = "\n\n"
        self.separatorCell = "|"
        self.separatorCellEscaped = "\\|"

        self.textCellEmpty = "\u200b"
        self.textCellSeparator = "---"

        self.labelPrefixObject = {
            "doc_title": "# ",
            "paragraph_title": "## "
        }

        self.labelPlaceholderObject = {
            "image": "![image]()",
            "chart": "![chart]()"
        }
