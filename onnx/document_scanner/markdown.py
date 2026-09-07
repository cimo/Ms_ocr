import sys
sys.dont_write_bytecode = True

import math
import unicodedata

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

                    if len(elementList) > 0:
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

                    if item["label"] != "table":
                        elementList = self._elementBoxCollect(scaleX, scaleY, item["coordinate"], page)

                        if len(elementList) > 0:
                            itemText = self._itemText(elementList, True, item["coordinate"][0] * scaleX, item["coordinate"][2] * scaleX)

                    if len(itemText) == 0:
                        pageText += f"[{item['label']}]\n" if len(pageText) == 0 else f"\n[{item['label']}]\n"
                    else:
                        pageText += f"{itemText}\n"

                secondaryText += f"- Page {astPage['number']}\n{pageText}\n"

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

        self.secondaryTitle = "SECONDARY ELEMENT"

class Docx:
    def _headingHash(self, level):
        return "#" * min(level, self.levelHeadingMax)

    def _rowText(self, cellList):
        result = "|"

        for a in range(len(cellList)):
            cellText = cellList[a].replace("|", "\\|").replace("\n", " ")

            result += f" {cellText} |"

        return result

    def execute(self, astPageList):
        result = ""

        for a in range(len(astPageList)):
            astPage = astPageList[a]

            isTableOpen = False

            for b in range(len(astPage["itemMainList"])):
                item = astPage["itemMainList"][b]

                if item["label"] == "tableRow":
                    cellList = item.get("cellList", [])

                    rowText = self._rowText(cellList)

                    if isTableOpen == False:
                        separatorText = "| --- " * len(cellList) + "|"

                        result += f"{rowText}\n{separatorText}\n"

                        isTableOpen = True
                    else:
                        result += f"{rowText}\n"

                    continue

                if isTableOpen:
                    result += "\n"

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
                result += "\n"

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
    def _columnLetter(self, index):
        result = ""

        value = index + 1

        while value > 0:
            remainder = (value - 1) % 26

            result = chr(65 + remainder) + result

            value = (value - 1) // 26

        return result

    def _cellEscape(self, text):
        return text.replace("|", "\\|").replace("\n", " ")

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

            if len(rowItemList) > 0:
                columnCount = len(rowItemList[0]["cellList"])

                headerText = "| row |"
                separatorText = "| --- |"

                for b in range(columnCount):
                    headerText += f" {self._columnLetter(b)} |"
                    separatorText += " --- |"

                result += f"{headerText}\n{separatorText}\n"

                for b in range(len(rowItemList)):
                    rowText = f"| {rowItemList[b]['number']} |"

                    for c in range(len(rowItemList[b]["cellList"])):
                        rowText += f" {self._cellEscape(rowItemList[b]['cellList'][c])} |"

                    result += f"{rowText}\n"

                result += "\n"

            mergeList = astPage.get("mergeList", [])

            if len(mergeList) > 0:
                result += f"Merge: {', '.join(mergeList)}\n\n"

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

    def _rowText(self, cellList):
        result = "|"

        for a in range(len(cellList)):
            cellText = cellList[a].replace("|", "\\|").replace("\n", " ")

            result += f" {cellText} |"

        return result

    def execute(self, astPageList):
        result = ""

        for a in range(len(astPageList)):
            astPage = astPageList[a]

            isTableOpen = False

            for b in range(len(astPage["itemMainList"])):
                item = astPage["itemMainList"][b]

                if item["label"] == "tableRow":
                    cellList = item.get("cellList", [])

                    rowText = self._rowText(cellList)

                    if isTableOpen == False:
                        separatorText = "| --- " * len(cellList) + "|"

                        result += f"{rowText}\n{separatorText}\n"

                        isTableOpen = True
                    else:
                        result += f"{rowText}\n"

                    continue

                if isTableOpen:
                    result += "\n"

                    isTableOpen = False

                if item["label"] == "doc_title":
                    result += f"# {item['text']}\n\n"
                elif item["label"] == "paragraph_title":
                    result += f"{self._headingHash(item['level'])} {item['text']}\n\n"
                else:
                    result += f"{item['text']}\n\n"

            if isTableOpen:
                result += "\n"

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
