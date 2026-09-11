import sys

sys.dont_write_bytecode = True

# Source
import test_office

class Markdown:
    def _textEscape(self, text):
        return text.replace("<", "\\<")

    def _cellEscape(self, text):
        return self._textEscape(text.replace(self.separatorCell, self.separatorCellEscaped).strip())

    def _rowWrite(self, textList):
        return f"{self.separatorCell} {f' {self.separatorCell} '.join(textList)} {self.separatorCell}"

    def _gridSize(self, cellList):
        countRow = 0
        countColumn = 0

        for a in range(len(cellList)):
            if cellList[a]["rowIndex"] + cellList[a]["rowSpan"] > countRow:
                countRow = cellList[a]["rowIndex"] + cellList[a]["rowSpan"]

            if cellList[a]["columnIndex"] + cellList[a]["columnSpan"] > countColumn:
                countColumn = cellList[a]["columnIndex"] + cellList[a]["columnSpan"]

        return {"countRow": countRow, "countColumn": countColumn}

    def _gridEmpty(self, countRow, countColumn):
        resultList = []

        for a in range(countRow):
            rowList = []

            for b in range(countColumn):
                rowList.append(self.textCellEmpty)

            resultList.append(rowList)

        return resultList

    def _secondaryAppend(self, blockList, secondaryList):
        if len(secondaryList) == 0:
            return blockList

        return blockList + ["> **SECONDARY ELEMENT**"] + secondaryList

    def _tableWrite(self, headerList, gridList):
        separatorList = []

        for a in range(len(headerList)):
            separatorList.append("---")

        lineList = [self._rowWrite(headerList), self._rowWrite(separatorList)]

        for a in range(len(gridList)):
            lineList.append(self._rowWrite(gridList[a]))

        return self.separatorLine.join(lineList)

    def execute(self, resultObject, extension):
        return self.builderObject[extension].execute(resultObject, extension)

    def __init__(self, extensionObject):
        self.separatorText = " "
        self.separatorLine = "\n"
        self.separatorBlock = "\n\n"
        self.separatorCell = "|"
        self.separatorCellEscaped = "\\|"
        
        self.textCellEmpty = "​"

        self.builderObject = {}

        builderImage = self.Image(self)

        for a in range(len(extensionObject["image"])):
            self.builderObject[extensionObject["image"][a]] = builderImage

        builderOffice = self.Office(self)

        for a in range(len(extensionObject["office"])):
            self.builderObject[extensionObject["office"][a]] = builderOffice

    class Image:
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

            return self.markdown._textEscape(self.markdown.separatorText.join(textList))

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

            return self.markdown.separatorText.join(textList)

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

            sizeObject = self.markdown._gridSize(cellList)

            gridList = self.markdown._gridEmpty(sizeObject["countRow"], sizeObject["countColumn"])

            rowRangeObject = self._rowRangeBuild(cellList)

            for a in range(len(cellList)):
                text = self.markdown._cellEscape(cellList[a]["text"])

                if len(text) == 0:
                    continue

                rowIndex = cellList[a]["rowIndex"]

                if cellList[a]["rowSpan"] > 1:
                    rowIndex = self._rowIndexAnchor(cellList[a], itemList, rowRangeObject)

                gridList[rowIndex][cellList[a]["columnIndex"]] = text

            headerList = []

            for a in range(sizeObject["countColumn"]):
                headerList.append(self.markdown.textCellEmpty)

            return self.markdown._tableWrite(headerList, gridList)

        def _blockWrite(self, layoutObject, tableList, itemList):
            label = layoutObject["label"]

            if label in self.labelPlaceholderList:
                return f"![{label}]({layoutObject['path']})"

            if label == "table":
                tableObject = self._tableFind(tableList, layoutObject)

                if tableObject != None:
                    return self._tableWrite(tableObject, itemList)

            text = self._textBuild(itemList, layoutObject["bbox"])

            if len(text) == 0:
                return ""

            prefix = self.labelPrefixObject[label] if label in self.labelPrefixObject else ""

            return f"{prefix}{text}"

        def execute(self, resultObject, extension):
            layoutList = resultObject["layoutList"]
            tableList = resultObject["tableList"]
            itemList = resultObject["itemList"]

            blockList = []
            secondaryList = []

            lineOrphanList = self._lineGroup(self._itemOrphanCollect(itemList, layoutList))

            indexOrphan = 0

            for a in range(len(layoutList)):
                while indexOrphan < len(lineOrphanList) and (lineOrphanList[indexOrphan]["page"], lineOrphanList[indexOrphan]["y1"]) < (layoutList[a]["page"], layoutList[a]["bbox"][1]):
                    blockList.append(self._lineText(lineOrphanList[indexOrphan]))

                    indexOrphan += 1

                block = self._blockWrite(layoutList[a], tableList, itemList)

                if len(block) == 0:
                    continue

                if layoutList[a]["flow"] == "secondary":
                    secondaryList.append(block)

                    continue

                blockList.append(block)

            while indexOrphan < len(lineOrphanList):
                blockList.append(self._lineText(lineOrphanList[indexOrphan]))

                indexOrphan += 1

            return self.markdown.separatorBlock.join(self.markdown._secondaryAppend(blockList, secondaryList))

        def __init__(self, markdown):
            self.levelLineOverlap = 0.5

            self.labelPrefixObject = {
                "doc_title": "# ",
                "paragraph_title": "## "
            }

            self.labelPlaceholderList = ["image", "chart"]

            self.markdown = markdown

    class Office:
        def _headingHash(self, level):
            return "#" * min(level, self.levelHeadingMax)

        def _tablePageCollect(self, tableList, numberPage):
            resultList = []

            for a in range(len(tableList)):
                if tableList[a]["page"] == numberPage:
                    resultList.append(tableList[a])

            return resultList

        def _gridFill(self, cellList):
            sizeObject = self.markdown._gridSize(cellList)

            resultList = self.markdown._gridEmpty(sizeObject["countRow"], sizeObject["countColumn"])

            for a in range(len(cellList)):
                text = self.markdown._cellEscape(cellList[a]["text"])

                if len(text) == 0:
                    continue

                resultList[cellList[a]["rowIndex"]][cellList[a]["columnIndex"]] = text

            return resultList

        def _tableWrite(self, tableObject):
            gridList = self._gridFill(tableObject["cellList"])

            headerList = []

            for a in range(len(gridList[0])):
                headerList.append(self.markdown.textCellEmpty)

            return self.markdown._tableWrite(headerList, gridList)

        def _sheetWrite(self, tableObject, astPage):
            gridList = self._gridFill(tableObject["cellList"])

            rowNumberList = []

            for a in range(len(astPage["itemMainList"])):
                if astPage["itemMainList"][a]["label"] == "tableRow":
                    rowNumberList.append(str(astPage["itemMainList"][a]["number"]))

            headerList = ["row"]

            for a in range(len(gridList[0])):
                headerList.append(self.office.gridColumnLetter(a))

            for a in range(len(gridList)):
                gridList[a].insert(0, rowNumberList[a])

            return self.markdown._tableWrite(headerList, gridList)

        def _blockBuild(self, itemList, tablePageList):
            resultList = []

            indexTable = 0
            isRow = False

            for a in range(len(itemList) + 1):
                item = itemList[a] if a < len(itemList) else None

                if item != None and item["label"] == "tableRow":
                    isRow = True

                    continue

                if isRow:
                    resultList.append(self._tableWrite(tablePageList[indexTable]))

                    indexTable += 1
                    isRow = False

                if item == None:
                    continue

                text = self.markdown._textEscape(item["text"])

                if item["label"] == "doc_title":
                    resultList.append(f"# {text}")
                elif item["label"] == "paragraph_title":
                    resultList.append(f"{self._headingHash(item['level'])} {text}")
                elif "isList" in item and item["isList"]:
                    resultList.append(f"- {text}")
                else:
                    resultList.append(text)

            return resultList

        def _secondaryBuild(self, itemList):
            resultList = []

            for a in range(len(itemList)):
                if "path" in itemList[a]:
                    resultList.append(f"![{itemList[a]['label']}]({itemList[a]['path']})")

                    continue

                text = self.markdown._textEscape(itemList[a]["text"])

                if len(text) == 0:
                    resultList.append(f"[{itemList[a]['label']}]")

                    continue

                if itemList[a]["label"] == "chart":
                    resultList.append(self._chartWrite(text))

                    continue

                resultList.append(text)

            return resultList

        def _chartWrite(self, text):
            textList = text.split(self.markdown.separatorLine)

            lineList = [textList[0]]

            for a in range(1, len(textList)):
                lineList.append(f"- {textList[a]}")

            return self.markdown.separatorLine.join(lineList)

        def _itemChildWrite(self, text):
            textList = text.split(self.markdown.separatorLine)

            lineList = [f"  - {textList[0]}"]

            for a in range(1, len(textList)):
                lineList.append(f"    {textList[a]}")

            return self.markdown.separatorLine.join(lineList)

        def _docxBuild(self, astPageList, tableList):
            blockList = []
            secondaryList = []

            for a in range(len(astPageList)):
                blockList = blockList + self._blockBuild(astPageList[a]["itemMainList"], self._tablePageCollect(tableList, astPageList[a]["number"]))
                secondaryList = secondaryList + self._secondaryBuild(astPageList[a]["itemSecondaryList"])

            return self.markdown.separatorBlock.join(self.markdown._secondaryAppend(blockList, secondaryList))

        def _xlsxBuild(self, astPageList, tableList):
            blockList = []
            secondaryList = []

            for a in range(len(astPageList)):
                astPage = astPageList[a]

                for b in range(len(astPage["itemMainList"])):
                    if astPage["itemMainList"][b]["label"] == "sheetName":
                        blockList.append(f"# {self.markdown._textEscape(astPage['itemMainList'][b]['text'])}")

                secondaryList = secondaryList + self._secondaryBuild(astPage["itemSecondaryList"])

                tablePageList = self._tablePageCollect(tableList, astPage["number"])

                for b in range(len(tablePageList)):
                    blockList.append(self._sheetWrite(tablePageList[b], astPage))

            return self.markdown.separatorBlock.join(self.markdown._secondaryAppend(blockList, secondaryList))

        def _pptxBuild(self, astPageList, tableList):
            blockList = []
            secondaryList = []

            for a in range(len(astPageList)):
                blockList = blockList + self._blockBuild(astPageList[a]["itemMainList"], self._tablePageCollect(tableList, astPageList[a]["number"]))

                itemSecondaryList = astPageList[a]["itemSecondaryList"]

                if len(itemSecondaryList) == 0:
                    continue

                lineList = [f"- Slide {astPageList[a]['number']}"]
                textList = self._secondaryBuild(itemSecondaryList)

                for b in range(len(textList)):
                    lineList.append(self._itemChildWrite(textList[b]))

                secondaryList.append(self.markdown.separatorLine.join(lineList))

            return self.markdown.separatorBlock.join(self.markdown._secondaryAppend(blockList, secondaryList))

        def execute(self, resultObject, extension):
            buildObject = {
                ".docx": self._docxBuild,
                ".xlsx": self._xlsxBuild,
                ".pptx": self._pptxBuild
            }

            return buildObject[extension](resultObject["astPageList"], resultObject["tableList"])

        def __init__(self, markdown):
            self.levelHeadingMax = 6

            self.markdown = markdown

            self.office = test_office.Office()
