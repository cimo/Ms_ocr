import sys
import math
import unicodedata

sys.dont_write_bytecode = True

# Source
import layout

class Page:
    def _medianFontSize(self, elementList):
        result = self.sizeDefault

        sizeList = []

        for a in range(len(elementList)):
            sizeList.append(elementList[a]["fontSize"])

        sizeList.sort()

        if len(sizeList) > 0 and sizeList[len(sizeList) // 2] > 0:
            result = sizeList[len(sizeList) // 2]

        return result

    def _titleSizeKey(self, elementList):
        return math.floor(self._medianFontSize(elementList) + 0.5)

    def _titleSizeRank(self, astPageList, pageObject):
        resultList = []

        for a in range(len(astPageList)):
            astPage = astPageList[a]

            if astPage["number"] in pageObject:
                page = pageObject[astPage["number"]]

                scaleX = page["width"] / astPage["imageWidth"]
                scaleY = page["height"] / astPage["imageHeight"]

                for b in range(len(astPage["itemMainList"])):
                    item = astPage["itemMainList"][b]

                    if item["label"] == "paragraph_title":
                        elementList = self._elementBoxCollect(scaleX, scaleY, item["coordinate"], page)

                        if len(elementList) > 0:
                            key = self._titleSizeKey(elementList)

                            if key not in resultList:
                                resultList.append(key)

        resultList.sort(reverse=True)

        return resultList

    def _headingHash(self, key, titleSizeRankList):
        level = self.levelHeadingBase

        if key in titleSizeRankList:
            level = self.levelHeadingBase + titleSizeRankList.index(key)

        return "#" * min(level, self.levelHeadingMax)

    def _lineGroup(self, elementList):
        resultList = []

        elementSortList = sorted(elementList, key=lambda element: element["y0"])

        for a in range(len(elementSortList)):
            element = elementSortList[a]

            isFound = False

            for b in range(len(resultList)):
                line = resultList[b]

                overlap = min(line["y1"], element["y1"]) - max(line["y0"], element["y0"])
                height = min(line["y1"] - line["y0"], element["y1"] - element["y0"])

                if height > 0 and overlap / height >= self.lineOverlapRatio:
                    line["elementList"].append(element)

                    line["y0"] = min(line["y0"], element["y0"])
                    line["y1"] = max(line["y1"], element["y1"])

                    isFound = True

                    break

            if isFound == False:
                resultList.append({"elementList": [element], "x0": 0, "x1": 0, "y0": element["y0"], "y1": element["y1"]})

        for a in range(len(resultList)):
            line = resultList[a]

            line["elementList"].sort(key=lambda element: element["x0"])

            line["x0"] = line["elementList"][0]["x0"]
            line["x1"] = line["elementList"][len(line["elementList"]) - 1]["x1"]

        resultList.sort(key=lambda line: line["y0"])

        return resultList

    def _wideCheck(self, character):
        return character != "" and unicodedata.east_asian_width(character) in self.wideList

    def _lineText(self, line, isPlain):
        result = ""

        for a in range(len(line["elementList"])):
            text = line["elementList"][a]["text"].strip()

            if len(text) > 0:
                if isPlain == False and line["elementList"][a]["isBold"] == True:
                    text = f"**{text}**"

                if len(result) == 0:
                    result = text
                elif self._wideCheck(result[-1:]) and self._wideCheck(text[0:1]):
                    result += text
                else:
                    result += f" {text}"

        return result

    def _elementBoxCollect(self, scaleX, scaleY, coordinate, page):
        resultList = []

        x1 = coordinate[0] * scaleX
        y1 = coordinate[1] * scaleY
        x2 = coordinate[2] * scaleX
        y2 = coordinate[3] * scaleY

        for a in range(len(page["elementList"])):
            element = page["elementList"][a]

            if element["type"] == "text":
                centerX = (element["x0"] + element["x1"]) / 2
                centerY = (element["y0"] + element["y1"]) / 2

                if centerX >= x1 and centerX <= x2 and centerY >= y1 and centerY <= y2:
                    resultList.append(element)

        return resultList

    def _elementAssign(self, itemList, page, scaleX, scaleY):
        resultList = []

        for a in range(len(itemList)):
            resultList.append([])

        for a in range(len(page["elementList"])):
            element = page["elementList"][a]

            if element["type"] == "text":
                centerX = (element["x0"] + element["x1"]) / 2
                centerY = (element["y0"] + element["y1"]) / 2

                bestIndex = -1
                bestArea = -1

                for b in range(len(itemList)):
                    coordinate = itemList[b]["coordinate"]

                    x1 = coordinate[0] * scaleX
                    y1 = coordinate[1] * scaleY
                    x2 = coordinate[2] * scaleX
                    y2 = coordinate[3] * scaleY

                    if centerX >= x1 and centerX <= x2 and centerY >= y1 and centerY <= y2:
                        area = (x2 - x1) * (y2 - y1)

                        if area > bestArea:
                            bestArea = area
                            bestIndex = b

                if bestIndex >= 0:
                    resultList[bestIndex].append(element)

        return resultList

    def _tableCellText(self, elementList):
        result = ""

        lineList = self._lineGroup(elementList)

        for a in range(len(lineList)):
            lineText = self._lineText(lineList[a], True)

            if len(lineText) == 0:
                continue

            if len(result) == 0:
                result = lineText
            elif self._wideCheck(result[-1:]) and self._wideCheck(lineText[0:1]):
                result += lineText
            else:
                result += f" {lineText}"

        return result

    def _tableSpanCheck(self, cellList):
        for a in range(len(cellList)):
            if cellList[a]["rowSpan"] > 1 or cellList[a]["columnSpan"] > 1:
                return True

        return False

    def _tableSlotAlign(self, elementList, slotX0, slotX1):
        x0 = elementList[0]["x0"]
        x1 = elementList[0]["x1"]

        for a in range(len(elementList)):
            x0 = min(x0, elementList[a]["x0"])
            x1 = max(x1, elementList[a]["x1"])

        gapLeft = x0 - slotX0
        gapRight = slotX1 - x1

        level = (slotX1 - slotX0) * self.levelAlign

        if gapLeft - gapRight > level:
            return "right"

        if gapRight - gapLeft > level:
            return "left"

        return "center"

    def _tableAlignList(self, columnCount, rowCount, alignObject):
        resultList = []

        for a in range(columnCount):
            countObject = {"left": 0, "right": 0, "center": 0}

            for b in range(rowCount):
                if (b, a) in alignObject:
                    countObject[alignObject[(b, a)]] += 1

            align = "left"

            for key in countObject:
                if countObject[key] > countObject[align]:
                    align = key

            resultList.append(align)

        return resultList

    def _tableCellFind(self, cellList, element, scaleX, scaleY):
        centerX = (element["x0"] + element["x1"]) / 2
        centerY = (element["y0"] + element["y1"]) / 2

        result = -1

        overlapMaximum = 0.0
        distanceMinimum = 0.0

        for a in range(len(cellList)):
            coordinate = cellList[a]["coordinate"]

            x0 = coordinate[0] * scaleX
            y0 = coordinate[1] * scaleY
            x1 = coordinate[2] * scaleX
            y1 = coordinate[3] * scaleY

            overlap = max(0.0, min(element["x1"], x1) - max(element["x0"], x0)) * max(0.0, min(element["y1"], y1) - max(element["y0"], y0))

            if overlap > overlapMaximum:
                result = a
                overlapMaximum = overlap

                continue

            if overlapMaximum > 0.0:
                continue

            distance = abs(centerX - (x0 + x1) / 2) + abs(centerY - (y0 + y1) / 2)

            if result == -1 or distance < distanceMinimum:
                result = a
                distanceMinimum = distance

        return result

    def _tableItemCreate(self, cell, cellIndex, elementList, scaleX):
        x0 = elementList[0]["x0"]
        x1 = elementList[0]["x1"]
        y0 = elementList[0]["y0"]
        y1 = elementList[0]["y1"]

        for a in range(len(elementList)):
            x0 = min(x0, elementList[a]["x0"])
            x1 = max(x1, elementList[a]["x1"])
            y0 = min(y0, elementList[a]["y0"])
            y1 = max(y1, elementList[a]["y1"])

        return {
            "x0": x0,
            "x1": x1,
            "y0": y0,
            "y1": y1,
            "cellIndex": cellIndex,
            "columnIndex": cell["columnIndex"],
            "columnSpan": cell["columnSpan"],
            "text": self._tableCellText(elementList),
            "align": self._tableSlotAlign(elementList, cell["coordinate"][0] * scaleX, cell["coordinate"][2] * scaleX)
        }

    def _tableTextVerticalCheck(self, lineList):
        if len(lineList) < self.countVerticalMinimum:
            return False

        for a in range(len(lineList)):
            if len(lineList[a]["elementList"]) != 1:
                return False

            if len(lineList[a]["elementList"][0]["text"].strip()) != 1:
                return False

        return True

    def _tableItemBuild(self, cellList, elementList, scaleX, scaleY):
        elementObject = {}

        for a in range(len(elementList)):
            index = self._tableCellFind(cellList, elementList[a], scaleX, scaleY)

            if index == -1:
                continue

            if index not in elementObject:
                elementObject[index] = []

            elementObject[index].append(elementList[a])

        resultList = []

        for index in elementObject:
            lineList = self._lineGroup(elementObject[index])

            if self._tableTextVerticalCheck(lineList):
                item = self._tableItemCreate(cellList[index], index, elementObject[index], scaleX)

                item["y0"] = lineList[0]["y0"]
                item["y1"] = lineList[0]["y1"]

                resultList.append(item)

                continue

            for a in range(len(lineList)):
                resultList.append(self._tableItemCreate(cellList[index], index, lineList[a]["elementList"], scaleX))

        return resultList

    def _tableCellHtml(self, item, alignList):
        attributeText = ""

        if item["columnSpan"] > 1:
            attributeText += f" colspan=\"{item['columnSpan']}\""

        if item["columnSpan"] == 1 and len(item["text"]) > 0 and alignList[item["columnIndex"]] != "left":
            attributeText += f" align=\"{alignList[item['columnIndex']]}\""

        return f"<td{attributeText}>{self._tableCellEscapeHtml(item['text'])}</td>"

    def _tableCellEscapeHtml(self, text):
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _tableCellEmptyHtml(self, columnSpan):
        if columnSpan > 1:
            return f"<td colspan=\"{columnSpan}\"></td>"

        return "<td></td>"

    def _tableDecorativeCheck(self, itemList):
        text = ""

        for a in range(len(itemList)):
            text += itemList[a]["text"]

        return layout.textDecorativeCheck(text)

    def _tableSectionIndex(self, itemList, columnCount):
        if columnCount < self.countColumnSection:
            return -1

        result = -1

        for a in range(len(itemList)):
            if len(itemList[a]["text"].strip()) == 0:
                continue

            if result >= 0:
                return -1

            result = a

        return result

    def _tableSectionHtml(self, item, columnCount):
        attributeText = ""

        if columnCount > 1:
            attributeText += f" colspan=\"{columnCount}\""

        if item["columnIndex"] > 0:
            if item["columnIndex"] + item["columnSpan"] >= columnCount:
                attributeText += " align=\"right\""
            else:
                attributeText += " align=\"center\""

        return f"<td{attributeText}>{self._tableCellEscapeHtml(item['text'])}</td>"

    def _tableTextHtml(self, rowList, columnCount, alignList):
        result = "<table>\n"

        for a in range(len(rowList)):
            itemSortList = sorted(rowList[a]["elementList"], key=lambda item: item["columnIndex"])

            if self._tableDecorativeCheck(itemSortList):
                continue

            indexSection = self._tableSectionIndex(itemSortList, columnCount)

            if indexSection >= 0:
                result += f"<tr>{self._tableSectionHtml(itemSortList[indexSection], columnCount)}</tr>\n"

                continue

            rowText = "<tr>"

            columnCurrent = 0

            for b in range(len(itemSortList)):
                item = itemSortList[b]

                if item["columnIndex"] > columnCurrent:
                    rowText += self._tableCellEmptyHtml(item["columnIndex"] - columnCurrent)

                    columnCurrent = item["columnIndex"]

                rowText += self._tableCellHtml(item, alignList)

                columnCurrent = max(columnCurrent, item["columnIndex"] + item["columnSpan"])

            if columnCount > columnCurrent:
                rowText += self._tableCellEmptyHtml(columnCount - columnCurrent)

            result += f"{rowText}</tr>\n"

        result += "</table>\n"

        return result

    def _tableColumnSeparatorList(self, cellList, indexList, columnStart, columnEnd):
        resultList = []

        for a in range(columnStart + 1, columnEnd):
            isSeparator = True

            for b in range(len(indexList)):
                cell = cellList[indexList[b]]

                if cell["columnIndex"] < a and cell["columnIndex"] + cell["columnSpan"] > a:
                    isSeparator = False

                    break

            if isSeparator:
                resultList.append(a)

        return resultList

    def _tableRowSeparatorList(self, cellList, indexList, rowStart, rowEnd):
        resultList = []

        for a in range(rowStart + 1, rowEnd):
            isSeparator = True

            for b in range(len(indexList)):
                cell = cellList[indexList[b]]

                if cell["rowIndex"] < a and cell["rowIndex"] + cell["rowSpan"] > a:
                    isSeparator = False

                    break

            if isSeparator:
                resultList.append(a)

        return resultList

    def _tableColumnBoundaryObject(self, cellList, indexList):
        resultObject = {}

        for a in range(len(indexList)):
            cell = cellList[indexList[a]]

            resultObject[cell["columnIndex"]] = True
            resultObject[cell["columnIndex"] + cell["columnSpan"]] = True

        return resultObject

    def _tableRowBoundaryObject(self, cellList, indexList):
        resultObject = {}

        for a in range(len(indexList)):
            cell = cellList[indexList[a]]

            resultObject[cell["rowIndex"]] = True
            resultObject[cell["rowIndex"] + cell["rowSpan"]] = True

        return resultObject

    def _tableSplitCheck(self, firstObject, secondObject):
        countShared = 0

        for key in firstObject:
            if key in secondObject:
                countShared += 1

        countTotal = len(firstObject) + len(secondObject) - countShared

        if countTotal == 0:
            return False

        return countShared / countTotal <= self.levelBlockSimilarity

    def _tableBlockSplit(self, cellList, indexList, columnStart, columnEnd, rowStart, rowEnd):
        separatorColumnList = self._tableColumnSeparatorList(cellList, indexList, columnStart, columnEnd)

        for a in range(len(separatorColumnList)):
            firstList = []
            secondList = []

            for b in range(len(indexList)):
                if cellList[indexList[b]]["columnIndex"] < separatorColumnList[a]:
                    firstList.append(indexList[b])
                else:
                    secondList.append(indexList[b])

            if len(firstList) == 0 or len(secondList) == 0:
                continue

            firstObject = self._tableRowBoundaryObject(cellList, firstList)
            secondObject = self._tableRowBoundaryObject(cellList, secondList)

            if self._tableSplitCheck(firstObject, secondObject) == False:
                continue

            return self._tableBlockSplit(cellList, firstList, columnStart, separatorColumnList[a], rowStart, rowEnd) + self._tableBlockSplit(
                cellList, secondList, separatorColumnList[a], columnEnd, rowStart, rowEnd
            )

        separatorRowList = self._tableRowSeparatorList(cellList, indexList, rowStart, rowEnd)

        for a in range(len(separatorRowList)):
            firstList = []
            secondList = []

            for b in range(len(indexList)):
                if cellList[indexList[b]]["rowIndex"] < separatorRowList[a]:
                    firstList.append(indexList[b])
                else:
                    secondList.append(indexList[b])

            if len(firstList) == 0 or len(secondList) == 0:
                continue

            firstObject = self._tableColumnBoundaryObject(cellList, firstList)
            secondObject = self._tableColumnBoundaryObject(cellList, secondList)

            if self._tableSplitCheck(firstObject, secondObject) == False:
                continue

            return self._tableBlockSplit(cellList, firstList, columnStart, columnEnd, rowStart, separatorRowList[a]) + self._tableBlockSplit(
                cellList, secondList, columnStart, columnEnd, separatorRowList[a], rowEnd
            )

        return [{"indexList": indexList, "columnStart": columnStart, "columnEnd": columnEnd}]

    def _tableBlockText(self, itemList, columnStart, columnEnd):
        itemBlockList = []

        for a in range(len(itemList)):
            itemBlockList.append({
                "x0": itemList[a]["x0"],
                "x1": itemList[a]["x1"],
                "y0": itemList[a]["y0"],
                "y1": itemList[a]["y1"],
                "columnIndex": itemList[a]["columnIndex"] - columnStart,
                "columnSpan": itemList[a]["columnSpan"],
                "text": itemList[a]["text"],
                "align": itemList[a]["align"]
            })

        rowList = self._lineGroup(itemBlockList)

        alignObject = {}

        for a in range(len(rowList)):
            for b in range(len(rowList[a]["elementList"])):
                item = rowList[a]["elementList"][b]

                alignObject[(a, item["columnIndex"])] = item["align"]

        columnCount = columnEnd - columnStart

        alignList = self._tableAlignList(columnCount, len(rowList), alignObject)

        return self._tableTextHtml(rowList, columnCount, alignList)

    def _tableSimpleCheck(self, tableObject):
        cellList = tableObject["cellList"]

        rowCount = tableObject["rowCount"]
        columnCount = tableObject["columnCount"]

        if rowCount == 0 or columnCount == 0 or len(cellList) == 0:
            return False

        countSpan = 0

        for a in range(len(cellList)):
            if cellList[a]["rowSpan"] > 1 or cellList[a]["columnSpan"] > 1:
                countSpan += 1

        if countSpan / float(len(cellList)) > self.levelCellSpan:
            return False

        return len(cellList) / float(rowCount * columnCount) >= self.levelCellFill

    def _tableGridCutRatio(self, tableObject, elementList, scaleX):
        if len(elementList) == 0:
            return 0.0

        countCut = 0

        for a in range(len(elementList)):
            element = elementList[a]

            margin = (element["x1"] - element["x0"]) * self.levelGridCutMargin

            for b in range(1, len(tableObject["edgeXList"]) - 1):
                edge = tableObject["edgeXList"][b] * scaleX

                if edge > element["x0"] + margin and edge < element["x1"] - margin:
                    countCut += 1

                    break

        return countCut / float(len(elementList))

    def _tableColumnInfer(self, tableObject, elementList, scaleX, coordinate, isReplace):
        lineList = []

        lineAllList = self._lineGroup(elementList)

        for a in range(len(lineAllList)):
            if len(lineAllList[a]["elementList"]) > 1:
                lineList.append(lineAllList[a])

        if len(lineList) < 2 or len(tableObject["edgeYList"]) < 2:
            return tableObject

        heightList = []

        for a in range(len(elementList)):
            heightList.append(elementList[a]["y1"] - elementList[a]["y0"])

        heightList.sort()

        height = heightList[len(heightList) // 2]

        tableX0 = coordinate[0] * scaleX
        tableX1 = coordinate[2] * scaleX

        binSize = height * self.levelColumnBin

        if height <= 0 or tableX1 - tableX0 <= binSize:
            return tableObject

        binCount = int((tableX1 - tableX0) / binSize) + 1

        countList = []

        for a in range(binCount):
            countList.append(0)

        for a in range(len(lineList)):
            binObject = {}

            for b in range(len(lineList[a]["elementList"])):
                element = lineList[a]["elementList"][b]

                binStart = max(0, int((element["x0"] - tableX0) / binSize))
                binEnd = min(binCount - 1, int((element["x1"] - tableX0) / binSize))

                for c in range(binStart, binEnd + 1):
                    binObject[c] = True

            for key in binObject:
                countList[key] += 1

        countMaximum = len(lineList) * (1 - self.levelColumnFree)

        boundaryList = []

        runStart = -1

        for a in range(binCount + 1):
            isFree = a < binCount and countList[a] <= countMaximum

            if isFree and runStart == -1:
                runStart = a
            elif isFree == False and runStart != -1:
                isInside = runStart > 0 and a < binCount

                if isInside and (a - runStart) * binSize >= height * self.levelColumnGap:
                    boundaryList.append(tableX0 + (runStart + a) / 2 * binSize)

                runStart = -1

        if isReplace == False and len(boundaryList) + 1 <= tableObject["columnCount"]:
            return tableObject

        edgeXList = [coordinate[0]]

        for a in range(len(boundaryList)):
            edgeXList.append(boundaryList[a] / scaleX)

        edgeXList.append(coordinate[2])

        edgeYList = tableObject["edgeYList"]

        cellList = []

        for a in range(len(edgeYList) - 1):
            for b in range(len(edgeXList) - 1):
                cellList.append({
                    "score": 1.0,
                    "coordinate": [edgeXList[b], edgeYList[a], edgeXList[b + 1], edgeYList[a + 1]],
                    "rowIndex": a,
                    "columnIndex": b,
                    "rowSpan": 1,
                    "columnSpan": 1
                })

        return {
            "rowCount": len(edgeYList) - 1,
            "columnCount": len(edgeXList) - 1,
            "edgeXList": edgeXList,
            "edgeYList": edgeYList,
            "cellList": cellList,
            "type": tableObject["type"]
        }

    def _tableText(self, tableObject, scaleX, scaleY, coordinate, page):
        # Complex table: sparse grid of merged cells, typical of a pdf form. Handled later with the pdf flow.
        if self._tableSimpleCheck(tableObject) == False:
            return ""

        elementList = self._elementBoxCollect(scaleX, scaleY, coordinate, page)

        isCut = self._tableGridCutRatio(tableObject, elementList, scaleX) >= self.levelGridCut

        if tableObject["type"] == "wireless" or isCut:
            tableObject = self._tableColumnInfer(tableObject, elementList, scaleX, coordinate, isCut)

        cellList = tableObject["cellList"]

        itemList = self._tableItemBuild(cellList, elementList, scaleX, scaleY)

        indexList = []

        for a in range(len(cellList)):
            indexList.append(a)

        blockList = self._tableBlockSplit(cellList, indexList, 0, tableObject["columnCount"], 0, tableObject["rowCount"])

        result = ""

        for a in range(len(blockList)):
            block = blockList[a]

            blockObject = {}

            for b in range(len(block["indexList"])):
                blockObject[block["indexList"][b]] = True

            itemBlockList = []

            for b in range(len(itemList)):
                if itemList[b]["cellIndex"] in blockObject:
                    itemBlockList.append(itemList[b])

            if len(itemBlockList) == 0:
                continue

            result += self._tableBlockText(itemBlockList, block["columnStart"], block["columnEnd"])

        return result

    def _itemText(self, elementList, isPlain, boxX0, referenceX1):
        result = ""

        lineList = self._lineGroup(elementList)

        fontSize = self._medianFontSize(elementList)

        maxX1 = 0
        minX0 = 0

        for a in range(len(lineList)):
            maxX1 = max(maxX1, lineList[a]["x1"])
            minX0 = lineList[a]["x0"] if a == 0 else min(minX0, lineList[a]["x0"])

        isMarker = len(lineList) > 1 and minX0 - boxX0 > fontSize * self.markerIndentRatio

        for a in range(len(lineList)):
            if abs(lineList[a]["x0"] - minX0) > fontSize * self.markerAlignRatio:
                isMarker = False

        openText = None

        for a in range(len(lineList)):
            lineText = self._lineText(lineList[a], isPlain)

            if a == 0:
                result = f"- {lineText}" if isMarker else lineText
            else:
                separator = " "

                if self._wideCheck(result[-1:]) and self._wideCheck(lineText[0:1]):
                    separator = ""

                if openText is not None and "://" in f"{openText}{lineText.split(')')[0]}":
                    separator = ""
                elif isPlain == False:
                    if lineList[a]["x0"] < lineList[a - 1]["x0"] - fontSize * self.markerAlignRatio:
                        separator = "\n"
                    elif isMarker:
                        if lineList[a - 1]["x1"] < referenceX1 - fontSize * self.lineBreakRatio:
                            separator = "\n- "
                    elif lineList[a - 1]["x1"] < maxX1 - fontSize * self.lineBreakRatio:
                        separator = "\n"

                result += f"{separator}{lineText}"

            for b in range(len(lineText)):
                if lineText[b] == "(":
                    openText = ""
                elif lineText[b] == ")":
                    openText = None
                elif openText is not None:
                    openText += lineText[b]

        return result

    def execute(self, astPageList, pageList):
        result = ""

        pageObject = {}

        for a in range(len(pageList)):
            pageObject[pageList[a]["number"]] = pageList[a]

        titleSizeRankList = self._titleSizeRank(astPageList, pageObject)

        for a in range(len(astPageList)):
            astPage = astPageList[a]

            if astPage["number"] in pageObject:
                page = pageObject[astPage["number"]]

                scaleX = page["width"] / astPage["imageWidth"]
                scaleY = page["height"] / astPage["imageHeight"]

                referenceX1 = 0

                for b in range(len(astPage["itemMainList"])):
                    referenceX1 = max(referenceX1, astPage["itemMainList"][b]["coordinate"][2] * scaleX)

                elementAssignList = self._elementAssign(astPage["itemMainList"], page, scaleX, scaleY)

                for b in range(len(astPage["itemMainList"])):
                    item = astPage["itemMainList"][b]

                    elementList = elementAssignList[b]

                    if item["label"] == "table":
                        tableText = self._tableText(item["tableObject"], scaleX, scaleY, item["coordinate"], page)

                        if len(tableText) > 0:
                            result += f"{tableText}\n"
                    elif len(elementList) > 0:
                        if item["label"] == "doc_title":
                            result += f"# {self._itemText(elementList, True, item['coordinate'][0] * scaleX, referenceX1)}\n\n"
                        elif item["label"] == "paragraph_title":
                            hashText = self._headingHash(self._titleSizeKey(elementList), titleSizeRankList)

                            result += f"{hashText} {self._itemText(elementList, True, item['coordinate'][0] * scaleX, referenceX1)}\n\n"
                        else:
                            result += f"{self._itemText(elementList, False, item['coordinate'][0] * scaleX, referenceX1)}\n\n"

        secondaryText = ""

        for a in range(len(astPageList)):
            astPage = astPageList[a]

            if astPage["number"] in pageObject and len(astPage["itemSecondaryList"]) > 0:
                page = pageObject[astPage["number"]]

                scaleX = page["width"] / astPage["imageWidth"]
                scaleY = page["height"] / astPage["imageHeight"]

                pageText = ""

                for b in range(len(astPage["itemSecondaryList"])):
                    item = astPage["itemSecondaryList"][b]

                    itemText = ""

                    if item["label"] == "table":
                        itemText = self._tableText(item["tableObject"], scaleX, scaleY, item["coordinate"], page)
                    else:
                        elementList = self._elementBoxCollect(scaleX, scaleY, item["coordinate"], page)

                        if len(elementList) > 0:
                            itemText = self._itemText(elementList, True, item["coordinate"][0] * scaleX, item["coordinate"][2] * scaleX)

                    if len(itemText) == 0:
                        pageText += f"[{item['label']}]\n" if len(pageText) == 0 else f"\n[{item['label']}]\n"
                    else:
                        pageText += f"{itemText}\n"

                pageLineList = pageText.splitlines()

                for b in range(len(pageLineList)):
                    if len(pageLineList[b]) > 0:
                        pageLineList[b] = f"  {pageLineList[b]}"

                secondaryText += f"- Page {astPage['number']}\n\n" + "\n".join(pageLineList) + "\n\n"

        if len(secondaryText) > 0:
            result += f"---\n\n{self.secondaryTitle}:\n\n{secondaryText}"

        return result

    def __init__(self):
        self.sizeDefault = 12

        self.levelHeadingBase = 2
        self.levelHeadingMax = 6

        self.lineOverlapRatio = 0.5
        self.markerIndentRatio = 0.8
        self.markerAlignRatio = 0.15
        self.lineBreakRatio = 4

        self.wideList = ["W", "F"]

        self.levelBlockSimilarity = 0.25
        self.countVerticalMinimum = 3
        self.levelAlign = 0.1
        self.levelCellFill = 0.5
        self.levelCellSpan = 0.5

        self.countColumnSection = 3

        self.levelColumnBin = 0.1
        self.levelColumnFree = 0.6
        self.levelColumnGap = 0.3
        self.levelGridCut = 0.1
        self.levelGridCutMargin = 0.15

        self.cellDefaultObject = {"rowSpan": 1, "columnSpan": 1}

        self.secondaryTitle = "SECONDARY ELEMENT"

class Docx:
    def _headingHash(self, level):
        return "#" * min(level, self.levelHeadingMax)

    def _tableCellEscapeHtml(self, text):
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _tableRowHtml(self, cellList):
        result = "<tr>"

        for a in range(len(cellList)):
            attributeText = ""

            if cellList[a]["rowSpan"] > 1:
                attributeText += f" rowspan=\"{cellList[a]['rowSpan']}\""

            if cellList[a]["columnSpan"] > 1:
                attributeText += f" colspan=\"{cellList[a]['columnSpan']}\""

            result += f"<td{attributeText}>{self._tableCellEscapeHtml(cellList[a]['text'])}</td>"

        return f"{result}</tr>"

    def execute(self, astPageList):
        result = ""

        for a in range(len(astPageList)):
            astPage = astPageList[a]

            isTableOpen = False

            for b in range(len(astPage["itemMainList"])):
                item = astPage["itemMainList"][b]

                if item["label"] == "tableRow":
                    cellList = item.get("cellList", [])

                    rowText = self._tableRowHtml(cellList)

                    if isTableOpen == False:
                        result += f"<table>\n{rowText}\n"

                        isTableOpen = True
                    else:
                        result += f"{rowText}\n"

                    continue

                if isTableOpen:
                    result += "</table>\n\n"

                    isTableOpen = False

                if item["label"] == "doc_title":
                    result += f"# {item['text']}\n\n"
                elif item["label"] == "paragraph_title":
                    result += f"{self._headingHash(item['level'])} {item['text']}\n\n"
                elif item.get("isList") == True:
                    result += f"- {item['text']}\n\n"
                else:
                    result += f"{item['text']}\n\n"

            if isTableOpen:
                result += "</table>\n\n"

        secondaryText = ""

        for a in range(len(astPageList)):
            astPage = astPageList[a]

            for b in range(len(astPage["itemSecondaryList"])):
                item = astPage["itemSecondaryList"][b]

                itemText = item["text"]

                if item["label"] == "chart" or len(item["text"]) == 0:
                    itemText = f"[{item['label']}]"

                secondaryText += f"{itemText}\n" if len(secondaryText) == 0 else f"\n{itemText}\n"

        if len(secondaryText) > 0:
            result += f"---\n\n{self.secondaryTitle}:\n\n{secondaryText}"

        return result

    def __init__(self):
        self.levelHeadingMax = 6

        self.secondaryTitle = "SECONDARY ELEMENT"

class Xlsx:
    def _tableColumnLetter(self, index):
        result = ""

        value = index + 1

        while value > 0:
            remainder = (value - 1) % 26

            result = chr(65 + remainder) + result

            value = (value - 1) // 26

        return result

    def _tableCellEscapeHtml(self, text):
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _tableColumnIndex(self, reference):
        result = 0

        for a in range(len(reference)):
            if reference[a].isalpha() == False:
                break

            result = result * 26 + (ord(reference[a].upper()) - 64)

        return result - 1

    def _tableRowNumber(self, reference):
        result = ""

        for a in range(len(reference)):
            if reference[a].isdigit():
                result += reference[a]

        return int(result)

    def _tableMergeObject(self, mergeList):
        resultObject = {}

        for a in range(len(mergeList)):
            referenceList = mergeList[a].split(":")

            if len(referenceList) != 2:
                continue

            columnStart = self._tableColumnIndex(referenceList[0])
            columnEnd = self._tableColumnIndex(referenceList[1])
            rowStart = self._tableRowNumber(referenceList[0])
            rowEnd = self._tableRowNumber(referenceList[1])

            for b in range(rowStart, rowEnd + 1):
                for c in range(columnStart, columnEnd + 1):
                    if b == rowStart and c == columnStart:
                        resultObject[(b, c)] = {"rowSpan": rowEnd - rowStart + 1, "columnSpan": columnEnd - columnStart + 1}
                    else:
                        resultObject[(b, c)] = None

        return resultObject

    def execute(self, astPageList):
        result = ""

        for a in range(len(astPageList)):
            astPage = astPageList[a]

            sheetName = ""
            rowItemList = []

            for b in range(len(astPage["itemMainList"])):
                item = astPage["itemMainList"][b]

                if item["label"] == "sheetName":
                    sheetName = item["text"]
                elif item["label"] == "tableRow":
                    rowItemList.append(item)

            result += f"# {sheetName}\n\n"

            mergeObject = self._tableMergeObject(astPage.get("mergeList", []))

            if len(rowItemList) > 0:
                columnCount = len(rowItemList[0]["cellList"])

                headerText = "<tr><td>row</td>"

                for b in range(columnCount):
                    headerText += f"<td>{self._tableColumnLetter(b)}</td>"

                result += f"<table>\n{headerText}</tr>\n"

                for b in range(len(rowItemList)):
                    rowNumber = rowItemList[b]["number"]

                    rowText = f"<tr><td>{rowNumber}</td>"

                    for c in range(len(rowItemList[b]["cellList"])):
                        mergeCellObject = mergeObject[(rowNumber, c)] if (rowNumber, c) in mergeObject else {"rowSpan": 1, "columnSpan": 1}

                        if mergeCellObject is None:
                            continue

                        attributeText = ""

                        if mergeCellObject["rowSpan"] > 1:
                            attributeText += f" rowspan=\"{mergeCellObject['rowSpan']}\""

                        if mergeCellObject["columnSpan"] > 1:
                            attributeText += f" colspan=\"{mergeCellObject['columnSpan']}\""

                        rowText += f"<td{attributeText}>{self._tableCellEscapeHtml(rowItemList[b]['cellList'][c])}</td>"

                    result += f"{rowText}</tr>\n"

                result += "</table>\n\n"

        secondaryText = ""

        for a in range(len(astPageList)):
            astPage = astPageList[a]

            for b in range(len(astPage["itemSecondaryList"])):
                item = astPage["itemSecondaryList"][b]

                itemText = item["text"]

                if item["label"] == "chart" or len(item["text"]) == 0:
                    itemText = f"[{item['label']}]"

                secondaryText += f"{itemText}\n" if len(secondaryText) == 0 else f"\n{itemText}\n"

        if len(secondaryText) > 0:
            result += f"---\n\n{self.secondaryTitle}:\n\n{secondaryText}"

        return result

    def __init__(self):
        self.secondaryTitle = "SECONDARY ELEMENT"

class Pptx:
    def _headingHash(self, level):
        return "#" * min(level, self.levelHeadingMax)

    def _tableCellEscapeHtml(self, text):
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _tableRowHtml(self, cellList):
        result = "<tr>"

        for a in range(len(cellList)):
            attributeText = ""

            if cellList[a]["rowSpan"] > 1:
                attributeText += f" rowspan=\"{cellList[a]['rowSpan']}\""

            if cellList[a]["columnSpan"] > 1:
                attributeText += f" colspan=\"{cellList[a]['columnSpan']}\""

            result += f"<td{attributeText}>{self._tableCellEscapeHtml(cellList[a]['text'])}</td>"

        return f"{result}</tr>"

    def execute(self, astPageList):
        result = ""

        for a in range(len(astPageList)):
            astPage = astPageList[a]

            isTableOpen = False

            for b in range(len(astPage["itemMainList"])):
                item = astPage["itemMainList"][b]

                if item["label"] == "tableRow":
                    cellList = item.get("cellList", [])

                    rowText = self._tableRowHtml(cellList)

                    if isTableOpen == False:
                        result += f"<table>\n{rowText}\n"

                        isTableOpen = True
                    else:
                        result += f"{rowText}\n"

                    continue

                if isTableOpen:
                    result += "</table>\n\n"

                    isTableOpen = False

                if item["label"] == "doc_title":
                    result += f"# {item['text']}\n\n"
                elif item["label"] == "paragraph_title":
                    result += f"{self._headingHash(item['level'])} {item['text']}\n\n"
                else:
                    result += f"{item['text']}\n\n"

            if isTableOpen:
                result += "</table>\n\n"

        secondaryText = ""

        for a in range(len(astPageList)):
            astPage = astPageList[a]

            if len(astPage["itemSecondaryList"]) > 0:
                pageText = ""

                for b in range(len(astPage["itemSecondaryList"])):
                    item = astPage["itemSecondaryList"][b]

                    itemText = item["text"]

                    if item["label"] == "chart" or len(item["text"]) == 0:
                        itemText = f"[{item['label']}]"

                    pageText += f"{itemText}\n" if len(pageText) == 0 else f"\n{itemText}\n"

                secondaryText += f"- Slide {astPage['number']}\n{pageText}\n"

        if len(secondaryText) > 0:
            result += f"---\n\n{self.secondaryTitle}:\n\n{secondaryText}"

        return result

    def __init__(self):
        self.levelHeadingMax = 6

        self.secondaryTitle = "SECONDARY ELEMENT"
