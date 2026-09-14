import sys
import icu

sys.dont_write_bytecode = True

# Source
import office

class Markdown:
    def _textJoin(self, textList):
        result = ""

        for a in range(len(textList)):
            if len(result) == 0:
                result = textList[a]

                continue

            if self._spacelessCheck(result[-1:]) and self._spacelessCheck(textList[a][0:1]):
                result += textList[a]

                continue

            result += f"{self.separatorText}{textList[a]}"

        return result

    def _spacelessCheck(self, character):
        if self._wideCheck(character):
            return True

        return icu.Char.getIntPropertyValue(character, icu.UProperty.LINE_BREAK) == self.lineBreakComplex

    def _wideCheck(self, character):
        if character == "":
            return False

        return icu.Char.getIntPropertyValue(character, icu.UProperty.EAST_ASIAN_WIDTH) in self.widthWideList

    def _textEscape(self, text):
        return text.replace("<", "\\<")

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

    def _cellEscape(self, text):
        return self._textEscape(text.replace(self.separatorCell, self.separatorCellEscaped).strip())

    def _tableWrite(self, headerList, gridList):
        separatorList = []

        for a in range(len(headerList)):
            separatorList.append("---")

        lineList = [self._rowWrite(headerList), self._rowWrite(separatorList)]

        for a in range(len(gridList)):
            lineList.append(self._rowWrite(gridList[a]))

        return self.separatorLine.join(lineList)

    def _rowWrite(self, textList):
        return f"{self.separatorCell} {f' {self.separatorCell} '.join(textList)} {self.separatorCell}"

    def _secondaryAppend(self, blockList, secondaryList):
        if len(secondaryList) == 0:
            return blockList

        return blockList + ["> **SECONDARY ELEMENT**"] + secondaryList

    def execute(self, resultObject, extension):
        return self.builderObject[extension].execute(resultObject, extension)

    def __init__(self, extensionObject):
        self.widthWideList = [icu.Char.getPropertyValueEnum(icu.UProperty.EAST_ASIAN_WIDTH, "W"), icu.Char.getPropertyValueEnum(icu.UProperty.EAST_ASIAN_WIDTH, "F")]
        self.lineBreakComplex = icu.Char.getPropertyValueEnum(icu.UProperty.LINE_BREAK, "SA")

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

        for a in range(len(extensionObject["pdf"])):
            self.builderObject[extensionObject["pdf"][a]] = builderImage

        builderOffice = self.Office(self)

        for a in range(len(extensionObject["office"])):
            self.builderObject[extensionObject["office"][a]] = builderOffice

    class Image:
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

        def _lineGroup(self, itemList, directionObject):
            resultList = []

            itemSortedList = sorted(itemList, key=lambda itemObject: (itemObject["page"], self._lineFlowKey(itemObject["bbox"], directionObject[itemObject["page"]])))

            for a in range(len(itemSortedList)):
                bboxList = itemSortedList[a]["bbox"]

                crossList = self._crossRange(bboxList, directionObject[itemSortedList[a]["page"]])

                isAdded = False

                for b in range(len(resultList)):
                    if resultList[b]["page"] != itemSortedList[a]["page"]:
                        continue

                    cross1 = max(crossList[0], resultList[b]["cross1"])
                    cross2 = min(crossList[1], resultList[b]["cross2"])

                    if cross2 <= cross1:
                        continue

                    if (cross2 - cross1) / float(min(crossList[1] - crossList[0], resultList[b]["cross2"] - resultList[b]["cross1"])) < self.levelLineOverlap:
                        continue

                    resultList[b]["cross1"] = min(resultList[b]["cross1"], crossList[0])
                    resultList[b]["cross2"] = max(resultList[b]["cross2"], crossList[1])

                    resultList[b]["bbox"] = self._bboxExpand(resultList[b]["bbox"], bboxList)

                    resultList[b]["itemList"].append(itemSortedList[a])

                    isAdded = True

                    break

                if isAdded == False:
                    resultList.append({"page": itemSortedList[a]["page"], "cross1": crossList[0], "cross2": crossList[1], "bbox": list(bboxList), "itemList": [itemSortedList[a]]})

            return sorted(resultList, key=lambda lineObject: (lineObject["page"], self._lineFlowKey(lineObject["bbox"], directionObject[lineObject["page"]])))

        def _lineFlowKey(self, bboxList, directionPageObject):
            if directionPageObject["isVertical"]:
                return -bboxList[2]

            return bboxList[1]

        def _crossRange(self, bboxList, directionPageObject):
            if directionPageObject["isVertical"]:
                return [bboxList[0], bboxList[2]]

            return [bboxList[1], bboxList[3]]

        def _columnRange(self, bboxList, directionPageObject):
            if directionPageObject["isVertical"]:
                return [bboxList[1], bboxList[3]]

            return [bboxList[0], bboxList[2]]

        def _bboxExpand(self, bboxList, bboxOtherList):
            return [
                min(bboxList[0], bboxOtherList[0]),
                min(bboxList[1], bboxOtherList[1]),
                max(bboxList[2], bboxOtherList[2]),
                max(bboxList[3], bboxOtherList[3])
            ]

        def _lineFlow(self, lineObject, layoutList, directionObject):
            directionPageObject = directionObject[lineObject["page"]]

            columnLineList = self._columnRange(lineObject["bbox"], directionPageObject)
            crossLineList = self._crossRange(lineObject["bbox"], directionPageObject)

            distanceBest = -1
            flowResult = "main"

            for a in range(len(layoutList)):
                if layoutList[a]["page"] != lineObject["page"]:
                    continue

                columnList = self._columnRange(layoutList[a]["bbox"], directionPageObject)

                if min(columnLineList[1], columnList[1]) <= max(columnLineList[0], columnList[0]):
                    continue

                crossList = self._crossRange(layoutList[a]["bbox"], directionPageObject)

                distance = max(crossList[0] - crossLineList[1], crossLineList[0] - crossList[1], 0)

                if distanceBest >= 0 and distance >= distanceBest:
                    continue

                distanceBest = distance
                flowResult = layoutList[a]["flow"]

            return flowResult

        def _lineText(self, lineObject, directionObject):
            directionPageObject = directionObject[lineObject["page"]]

            itemSortedList = sorted(lineObject["itemList"], key=lambda itemObject: self._itemOrderKey(itemObject["bbox"], directionPageObject))

            result = ""

            for a in range(len(itemSortedList)):
                if len(result) == 0:
                    result = itemSortedList[a]["text"]

                    continue

                if self._spaceCheck(itemSortedList[a - 1]["bbox"], itemSortedList[a]["bbox"], result, itemSortedList[a]["text"], directionPageObject):
                    result += self.markdown.separatorText

                result += itemSortedList[a]["text"]

            return self.markdown._textEscape(result)

        def _spaceCheck(self, bboxPreviousList, bboxList, textPrevious, text, directionPageObject):
            if directionPageObject["isVertical"]:
                gap = bboxList[1] - bboxPreviousList[3]
                size = min(bboxPreviousList[2] - bboxPreviousList[0], bboxList[2] - bboxList[0])
            elif directionPageObject["isRightToLeft"]:
                gap = bboxPreviousList[0] - bboxList[2]
                size = min(bboxPreviousList[3] - bboxPreviousList[1], bboxList[3] - bboxList[1])
            else:
                gap = bboxList[0] - bboxPreviousList[2]
                size = min(bboxPreviousList[3] - bboxPreviousList[1], bboxList[3] - bboxList[1])

            if self.markdown._wideCheck(textPrevious[-1:]) and self.markdown._wideCheck(text[0:1]):
                return False

            return gap >= size * self.levelSpaceGap

        def _itemOrderKey(self, bboxList, directionPageObject):
            if directionPageObject["isVertical"]:
                return bboxList[1]

            if directionPageObject["isRightToLeft"]:
                return -bboxList[2]

            return bboxList[0]

        def _blockWrite(self, layoutObject, tableList, itemList, directionObject):
            label = layoutObject["label"]

            if label in self.labelPlaceholderList:
                return f"![{label}]({layoutObject['path']})"

            if label == "table":
                tableObject = self._tableFind(tableList, layoutObject)

                if tableObject != None:
                    return self._tableWrite(tableObject, itemList, directionObject)

            text = self._textBuild(itemList, layoutObject["page"], layoutObject["bbox"], directionObject)

            if len(text) == 0:
                return ""

            prefix = self.labelPrefixObject[label] if label in self.labelPrefixObject else ""

            return f"{prefix}{text}"

        def _tableFind(self, tableList, layoutObject):
            for a in range(len(tableList)):
                if tableList[a]["page"] != layoutObject["page"]:
                    continue

                if tableList[a]["bbox"] != layoutObject["bbox"]:
                    continue

                return tableList[a]

            return None

        def _tableWrite(self, tableObject, itemList, directionObject):
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
                    rowIndex = self._rowIndexAnchor(cellList[a], itemList, tableObject["page"], rowRangeObject)

                gridList[rowIndex][cellList[a]["columnIndex"]] = text

            directionPageObject = directionObject[tableObject["page"]]

            if directionPageObject["isVertical"] or directionPageObject["isRightToLeft"]:
                for a in range(len(gridList)):
                    gridList[a].reverse()

            headerList = []

            for a in range(sizeObject["countColumn"]):
                headerList.append(self.markdown.textCellEmpty)

            return self.markdown._tableWrite(headerList, gridList)

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

        def _rowIndexAnchor(self, cellObject, itemList, numberPage, rowRangeObject):
            bboxList = cellObject["bbox"]

            y1List = []
            y2List = []

            for a in range(len(itemList)):
                if itemList[a]["page"] != numberPage:
                    continue

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

        def _textBuild(self, itemList, numberPage, bboxList, directionObject):
            itemInsideList = []

            for a in range(len(itemList)):
                if itemList[a]["page"] != numberPage:
                    continue

                centerPointObject = itemList[a]["centerPoint"]

                if centerPointObject["x"] < bboxList[0] or centerPointObject["x"] > bboxList[2]:
                    continue

                if centerPointObject["y"] < bboxList[1] or centerPointObject["y"] > bboxList[3]:
                    continue

                itemInsideList.append(itemList[a])

            lineList = self._lineGroup(itemInsideList, directionObject)

            textList = []

            for a in range(len(lineList)):
                textList.append(self._lineText(lineList[a], directionObject))

            return self.markdown._textJoin(textList)

        def _blockJoinable(self, layoutObject):
            label = layoutObject["label"]

            if label in self.labelPlaceholderList or label in self.labelBarrierList:
                return False

            return label not in self.labelPrefixObject

        def _blockMerge(self, blockList, directionObject):
            resultList = []

            for a in range(len(blockList)):
                if len(resultList) > 0:
                    blockPrevious = resultList[len(resultList) - 1]

                    if blockPrevious["isJoinable"] and blockList[a]["isJoinable"] and self._pageBreakCheck(blockPrevious, blockList[a], directionObject) and self._sentenceEndCheck(blockPrevious["text"]) == False:
                        blockPrevious["text"] = self.markdown._textJoin([blockPrevious["text"], blockList[a]["text"]])
                        blockPrevious["page"] = blockList[a]["page"]
                        blockPrevious["bbox"] = blockList[a]["bbox"]

                        continue

                resultList.append(blockList[a])

            textList = []

            for a in range(len(resultList)):
                textList.append(resultList[a]["text"])

            return textList

        def _pageBreakCheck(self, blockPrevious, blockObject, directionObject):
            if blockPrevious["page"] + 1 != blockObject["page"]:
                return False

            columnPreviousList = self._columnRange(blockPrevious["bbox"], directionObject[blockPrevious["page"]])
            columnList = self._columnRange(blockObject["bbox"], directionObject[blockObject["page"]])

            column1 = max(columnPreviousList[0], columnList[0])
            column2 = min(columnPreviousList[1], columnList[1])

            if column2 <= column1:
                return False

            sizeMinimum = min(columnPreviousList[1] - columnPreviousList[0], columnList[1] - columnList[0])

            return (column2 - column1) / float(sizeMinimum) >= self.levelColumnOverlap

        def _sentenceEndCheck(self, text):
            textClean = self._sentenceTailStrip(text)

            return len(textClean) > 0 and icu.Char.hasBinaryProperty(textClean[-1:], icu.UProperty.S_TERM)

        def _sentenceTailStrip(self, text):
            result = text.strip()

            while len(result) > 0:
                character = result[-1:]

                if icu.Char.charType(character) == icu.UCharCategory.END_PUNCTUATION:
                    indexOpen = self._groupOpenIndex(result)

                    result = result[0:indexOpen] if indexOpen >= 0 else result[0:-1]

                    continue

                if icu.Char.charType(character) == icu.UCharCategory.FINAL_PUNCTUATION or icu.Char.hasBinaryProperty(character, icu.UProperty.QUOTATION_MARK) or icu.Char.isUWhiteSpace(character):
                    result = result[0:-1]

                    continue

                break

            return result

        def _groupOpenIndex(self, text):
            for a in range(len(text) - 2, len(text) - 2 - self.levelReferenceLength, -1):
                if a < 0:
                    break

                character = text[a]

                if icu.Char.charType(character) == icu.UCharCategory.START_PUNCTUATION:
                    return a

                if icu.Char.charType(character) == icu.UCharCategory.END_PUNCTUATION or icu.Char.hasBinaryProperty(character, icu.UProperty.S_TERM):
                    break

            return -1

        def execute(self, resultObject, extension):
            layoutList = resultObject["layoutList"]
            tableList = resultObject["tableList"]
            itemList = resultObject["itemList"]

            directionObject = {}

            for a in range(len(resultObject["directionList"])):
                directionObject[resultObject["directionList"][a]["page"]] = resultObject["directionList"][a]

            blockList = []
            secondaryList = []

            lineOrphanList = self._lineGroup(self._itemOrphanCollect(itemList, layoutList), directionObject)

            indexOrphan = 0

            for a in range(len(layoutList)):
                flowBlock = self._lineFlowKey(layoutList[a]["bbox"], directionObject[layoutList[a]["page"]])

                while indexOrphan < len(lineOrphanList) and (lineOrphanList[indexOrphan]["page"], self._lineFlowKey(lineOrphanList[indexOrphan]["bbox"], directionObject[lineOrphanList[indexOrphan]["page"]])) < (layoutList[a]["page"], flowBlock):
                    if self._lineFlow(lineOrphanList[indexOrphan], layoutList, directionObject) == "main":
                        blockList.append({"text": self._lineText(lineOrphanList[indexOrphan], directionObject), "isJoinable": False, "page": lineOrphanList[indexOrphan]["page"], "bbox": None})
                    else:
                        secondaryList.append(self._lineText(lineOrphanList[indexOrphan], directionObject))

                    indexOrphan += 1

                block = self._blockWrite(layoutList[a], tableList, itemList, directionObject)

                if len(block) == 0:
                    continue

                if layoutList[a]["flow"] == "secondary":
                    secondaryList.append(block)

                    continue

                blockList.append({"text": block, "isJoinable": self._blockJoinable(layoutList[a]), "page": layoutList[a]["page"], "bbox": layoutList[a]["bbox"]})

            while indexOrphan < len(lineOrphanList):
                if self._lineFlow(lineOrphanList[indexOrphan], layoutList, directionObject) == "main":
                    blockList.append({"text": self._lineText(lineOrphanList[indexOrphan], directionObject), "isJoinable": False, "page": lineOrphanList[indexOrphan]["page"], "bbox": None})
                else:
                    secondaryList.append(self._lineText(lineOrphanList[indexOrphan], directionObject))

                indexOrphan += 1

            return self.markdown.separatorBlock.join(self.markdown._secondaryAppend(self._blockMerge(blockList, directionObject), secondaryList))

        def __init__(self, markdown):
            self.levelLineOverlap = 0.5
            self.levelColumnOverlap = 0.5
            self.levelSpaceGap = 0.15

            self.levelReferenceLength = 20

            self.labelPrefixObject = {
                "doc_title": "# ",
                "paragraph_title": "## "
            }

            self.labelPlaceholderList = ["image", "chart"]

            self.labelBarrierList = ["table", "header", "footer"]

            self.markdown = markdown

    class Office:
        def _docxBuild(self, astPageList, tableList):
            blockList = []
            secondaryList = []

            for a in range(len(astPageList)):
                blockList = blockList + self._blockBuild(astPageList[a]["itemMainList"], self._tablePageCollect(tableList, astPageList[a]["number"]))
                secondaryList = secondaryList + self._secondaryBuild(astPageList[a]["itemSecondaryList"])

            return self.markdown.separatorBlock.join(self.markdown._secondaryAppend(blockList, secondaryList))

        def _tablePageCollect(self, tableList, numberPage):
            resultList = []

            for a in range(len(tableList)):
                if tableList[a]["page"] == numberPage:
                    resultList.append(tableList[a])

            return resultList

        def _blockBuild(self, itemList, tablePageList):
            resultList = []

            indexTable = 0
            isRow = False
            directionObject = None

            for a in range(len(itemList) + 1):
                item = itemList[a] if a < len(itemList) else None

                if item != None and item["label"] == "tableRow":
                    if isRow == False:
                        directionObject = item["direction"]

                    isRow = True

                    continue

                if isRow:
                    resultList.append(self._tableWrite(tablePageList[indexTable], directionObject))

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

        def _tableWrite(self, tableObject, directionObject):
            gridList = self._gridFill(tableObject["cellList"])

            if directionObject["isVertical"] or directionObject["isRightToLeft"]:
                for a in range(len(gridList)):
                    gridList[a].reverse()

            headerList = []

            for a in range(len(gridList[0])):
                headerList.append(self.markdown.textCellEmpty)

            return self.markdown._tableWrite(headerList, gridList)

        def _gridFill(self, cellList):
            sizeObject = self.markdown._gridSize(cellList)

            resultList = self.markdown._gridEmpty(sizeObject["countRow"], sizeObject["countColumn"])

            for a in range(len(cellList)):
                text = self.markdown._cellEscape(cellList[a]["text"])

                if len(text) == 0:
                    continue

                resultList[cellList[a]["rowIndex"]][cellList[a]["columnIndex"]] = text

            return resultList

        def _headingHash(self, level):
            return "#" * min(level, self.levelHeadingMax)

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

        def _sheetWrite(self, tableObject, astPage):
            gridList = self._gridFill(tableObject["cellList"])

            rowNumberList = []

            for a in range(len(astPage["itemMainList"])):
                if astPage["itemMainList"][a]["label"] == "tableRow":
                    rowNumberList.append(str(astPage["itemMainList"][a]["number"]))

            columnList = []

            for a in range(len(gridList[0])):
                columnList.append(self.office.gridColumnLetter(a))

            if astPage["direction"]["isRightToLeft"]:
                columnList.reverse()

                for a in range(len(gridList)):
                    gridList[a].reverse()

            headerList = ["row"] + columnList

            for a in range(len(gridList)):
                gridList[a].insert(0, rowNumberList[a])

            return self.markdown._tableWrite(headerList, gridList)

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

        def _itemChildWrite(self, text):
            textList = text.split(self.markdown.separatorLine)

            lineList = [f"  - {textList[0]}"]

            for a in range(1, len(textList)):
                lineList.append(f"    {textList[a]}")

            return self.markdown.separatorLine.join(lineList)

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

            self.office = office.Office()
