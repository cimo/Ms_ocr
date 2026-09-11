import sys
import os
import re
import json
import zipfile
import datetime
import unicodedata
import xml.etree.ElementTree

sys.dont_write_bytecode = True

class Office:
    def _xmlNodeTag(self, node):
        return node.tag.split("}")[1] if "}" in node.tag else node.tag

    def _xmlNodeValue(self, node, namespace):
        return node.attrib.get(f"{{{namespace}}}val", "")

    def _xmlRootBuild(self, pathFile, zipFile):
        result = None

        if pathFile in zipFile.namelist():
            result = xml.etree.ElementTree.fromstring(zipFile.read(pathFile))

        return result

    def _xmlChartText(self, chartRootNode):
        namespaceChart = "http://schemas.openxmlformats.org/drawingml/2006/chart"
        namespaceDrawing = "http://schemas.openxmlformats.org/drawingml/2006/main"

        result = "chart"

        plotAreaNode = chartRootNode.find(f".//{{{namespaceChart}}}plotArea")

        if plotAreaNode is not None:
            for node in plotAreaNode:
                if self._xmlNodeTag(node).endswith("Chart"):
                    result = self._xmlNodeTag(node)

                    break

        titleText = ""

        titleNode = chartRootNode.find(f".//{{{namespaceChart}}}title")

        if titleNode is not None:
            for node in titleNode.iter(f"{{{namespaceDrawing}}}t"):
                titleText += node.text if node.text is not None else ""

        if titleText != "":
            result += f" - {titleText}"

        for serieNode in chartRootNode.iter(f"{{{namespaceChart}}}ser"):
            nameText = ""

            textNode = serieNode.find(f"{{{namespaceChart}}}tx")

            if textNode is not None:
                for node in textNode.iter(f"{{{namespaceChart}}}v"):
                    if node.text is not None:
                        nameText = node.text

                    break

            categoryObject = {}

            categoryNode = serieNode.find(f"{{{namespaceChart}}}cat")

            if categoryNode is not None:
                for node in categoryNode.iter(f"{{{namespaceChart}}}pt"):
                    valueNode = node.find(f"{{{namespaceChart}}}v")

                    if valueNode is not None and valueNode.text is not None:
                        categoryObject[node.attrib.get("idx", "")] = valueNode.text

            pairList = []

            valueParentNode = serieNode.find(f"{{{namespaceChart}}}val")

            if valueParentNode is not None:
                for node in valueParentNode.iter(f"{{{namespaceChart}}}pt"):
                    valueNode = node.find(f"{{{namespaceChart}}}v")

                    if valueNode is not None and valueNode.text is not None:
                        index = node.attrib.get("idx", "")

                        categoryText = categoryObject[index] if index in categoryObject else index

                        pairList.append(f"{categoryText}={valueNode.text}")

            serieText = ", ".join(pairList)

            if nameText != "":
                result += f"\n{nameText}: {serieText}"
            elif serieText != "":
                result += f"\n{serieText}"

        return result

    def _layoutBuild(self, astPageList):
        resultList = []

        for a in range(len(astPageList)):
            flowObject = {"main": astPageList[a]["itemMainList"], "secondary": astPageList[a]["itemSecondaryList"]}

            for flow in flowObject:
                itemList = flowObject[flow]

                for b in range(len(itemList)):
                    resultList.append({
                        "id": len(resultList) + 1,
                        "page": astPageList[a]["number"],
                        "flow": flow,
                        "label": itemList[b]["label"],
                        "score": 0.0,
                        "bbox": list([0, 0, 0, 0]),
                        "centerPoint": dict({"x": 0, "y": 0}),
                        "path": itemList[b]["path"] if "path" in itemList[b] else ""
                    })

        return resultList

    def _tableBuild(self, astPageList):
        resultList = []

        for a in range(len(astPageList)):
            itemList = astPageList[a]["itemMainList"]

            rowList = []

            for b in range(len(itemList) + 1):
                if b < len(itemList) and itemList[b]["label"] == "tableRow":
                    rowList.append(itemList[b])

                    continue

                if len(rowList) == 0:
                    continue

                cellList = []

                gridList = self._gridBuild(rowList)

                for c in range(len(gridList)):
                    cellList.append({
                        "rowIndex": gridList[c]["rowIndex"],
                        "columnIndex": gridList[c]["columnIndex"],
                        "rowSpan": gridList[c]["rowSpan"],
                        "columnSpan": gridList[c]["columnSpan"],
                        "bbox": list([0, 0, 0, 0]),
                        "centerPoint": dict({"x": 0, "y": 0}),
                        "text": gridList[c]["text"]
                    })

                resultList.append({
                    "id": len(resultList) + 1,
                    "page": astPageList[a]["number"],
                    "type": "office",
                    "bbox": list([0, 0, 0, 0]),
                    "centerPoint": dict({"x": 0, "y": 0}),
                    "cellList": cellList
                })

                rowList = []

        return resultList

    def _tableXlsxBuild(self, astPageList):
        resultList = []

        for a in range(len(astPageList)):
            astPage = astPageList[a]

            rowItemList = []

            for b in range(len(astPage["itemMainList"])):
                if astPage["itemMainList"][b]["label"] == "tableRow":
                    rowItemList.append(astPage["itemMainList"][b])

            if len(rowItemList) == 0:
                continue

            mergeObject = self._gridMergeObject(astPage["mergeList"] if "mergeList" in astPage else [])

            cellList = []

            for b in range(len(rowItemList)):
                rowNumber = rowItemList[b]["number"]

                for c in range(len(rowItemList[b]["cellList"])):
                    spanObject = mergeObject[(rowNumber, c)] if (rowNumber, c) in mergeObject else {"rowSpan": 1, "columnSpan": 1}

                    if spanObject is None:
                        continue

                    cellList.append({
                        "rowIndex": b,
                        "columnIndex": c,
                        "rowSpan": spanObject["rowSpan"],
                        "columnSpan": spanObject["columnSpan"],
                        "bbox": list([0, 0, 0, 0]),
                        "centerPoint": dict({"x": 0, "y": 0}),
                        "text": rowItemList[b]["cellList"][c]
                    })

            resultList.append({
                "id": len(resultList) + 1,
                "page": astPage["number"],
                "type": "office",
                "bbox": list([0, 0, 0, 0]),
                "centerPoint": dict({"x": 0, "y": 0}),
                "cellList": cellList
            })

        return resultList

    def _itemBuild(self, astPageList):
        resultList = []

        for a in range(len(astPageList)):
            itemList = astPageList[a]["itemMainList"] + astPageList[a]["itemSecondaryList"]

            for b in range(len(itemList)):
                if len(itemList[b]["text"]) == 0:
                    continue

                resultList.append({
                    "id": len(resultList) + 1,
                    "page": astPageList[a]["number"],
                    "bbox": list([0, 0, 0, 0]),
                    "centerPoint": dict({"x": 0, "y": 0}),
                    "text": itemList[b]["text"],
                    "isMatch": False
                })

        return resultList

    def _gridColumnIndex(self, reference):
        result = 0

        for a in range(len(reference)):
            if reference[a].isalpha() == False:
                break

            result = result * 26 + (ord(reference[a].upper()) - 64)

        return result - 1

    def _gridRowNumber(self, reference):
        result = ""

        for a in range(len(reference)):
            if reference[a].isdigit():
                result += reference[a]

        return int(result)

    def _gridMergeObject(self, mergeList):
        resultObject = {}

        for a in range(len(mergeList)):
            referenceList = mergeList[a].split(":")

            if len(referenceList) != 2:
                continue

            columnStart = self._gridColumnIndex(referenceList[0])
            columnEnd = self._gridColumnIndex(referenceList[1])
            rowStart = self._gridRowNumber(referenceList[0])
            rowEnd = self._gridRowNumber(referenceList[1])

            for b in range(rowStart, rowEnd + 1):
                for c in range(columnStart, columnEnd + 1):
                    if b == rowStart and c == columnStart:
                        resultObject[(b, c)] = {"rowSpan": rowEnd - rowStart + 1, "columnSpan": columnEnd - columnStart + 1}
                    else:
                        resultObject[(b, c)] = None

        return resultObject

    def _gridBuild(self, rowList):
        occupiedObject = {}

        resultList = []

        for a in range(len(rowList)):
            cellList = rowList[a]["cellList"]

            columnIndex = 0

            for b in range(len(cellList)):
                while (a, columnIndex) in occupiedObject:
                    columnIndex += 1

                rowSpan = cellList[b]["rowSpan"]
                columnSpan = cellList[b]["columnSpan"]

                for c in range(rowSpan):
                    for d in range(columnSpan):
                        occupiedObject[(a + c, columnIndex + d)] = True

                resultList.append({
                    "rowIndex": a,
                    "columnIndex": columnIndex,
                    "rowSpan": rowSpan,
                    "columnSpan": columnSpan,
                    "text": cellList[b]["text"]
                })

                columnIndex += columnSpan

        return resultList

    def gridColumnLetter(self, index):
        result = ""

        value = index + 1

        while value > 0:
            remainder = (value - 1) % 26

            result = chr(65 + remainder) + result

            value = (value - 1) // 26

        return result

    def execute(self, pathInput, pathOutput, extension):
        astPageList = self.readerObject[extension].execute(pathInput, pathOutput)["pageList"]

        resultObject = {
            "pageCount": len(astPageList),
            "astPageList": astPageList,
            "layoutList": self._layoutBuild(astPageList),
            "tableList": self._tableXlsxBuild(astPageList) if extension == ".xlsx" else self._tableBuild(astPageList),
            "itemList": self._itemBuild(astPageList)
        }

        return resultObject

    class Docx:
        def _fallbackRemove(self, node):
            for child in list(node):
                if self.office._xmlNodeTag(child) == "Fallback":
                    node.remove(child)
                else:
                    self._fallbackRemove(child)

        def _textCollect(self, node):
            result = ""

            tag = self.office._xmlNodeTag(node)

            if tag != "txbxContent":
                if tag == "t":
                    result += node.text if node.text is not None else ""
                elif tag == "tab" or tag == "br":
                    result += " "

                for childNode in node:
                    result += self._textCollect(childNode)

            return result

        def _paragraphText(self, paragraphNode):
            return self._textCollect(paragraphNode).strip()

        def _paragraphSize(self, sizeDocument, paragraphNode):
            sizeDefault = sizeDocument

            sizeDefaultNode = paragraphNode.find(f"{{{self.namespaceMain}}}pPr/{{{self.namespaceMain}}}rPr/{{{self.namespaceMain}}}sz")

            if sizeDefaultNode is not None and self.office._xmlNodeValue(sizeDefaultNode, self.namespaceMain) != "":
                sizeDefault = float(self.office._xmlNodeValue(sizeDefaultNode, self.namespaceMain))

            countObject = {}

            for runNode in paragraphNode.iter(f"{{{self.namespaceMain}}}r"):
                size = sizeDefault
                length = 0

                sizeNode = runNode.find(f"{{{self.namespaceMain}}}rPr/{{{self.namespaceMain}}}sz")

                if sizeNode is not None and self.office._xmlNodeValue(sizeNode, self.namespaceMain) != "":
                    size = float(self.office._xmlNodeValue(sizeNode, self.namespaceMain))

                for node in runNode.iter(f"{{{self.namespaceMain}}}t"):
                    if node.text is not None:
                        length += len(node.text)

                if length > 0:
                    countObject[size] = countObject.get(size, 0) + length

            result = 0.0
            countMax = 0

            for size in countObject:
                if countObject[size] > countMax:
                    countMax = countObject[size]
                    result = size

            return result

        def _paragraphStyle(self, paragraphNode):
            result = ""

            for node in paragraphNode.iter(f"{{{self.namespaceMain}}}pStyle"):
                result = self.office._xmlNodeValue(node, self.namespaceMain)

            return result

        def _paragraphOutlineLevel(self, paragraphNode):
            result = -1

            for node in paragraphNode.iter(f"{{{self.namespaceMain}}}outlineLvl"):
                if self.office._xmlNodeValue(node, self.namespaceMain) != "":
                    result = int(self.office._xmlNodeValue(node, self.namespaceMain))

            return result

        def _paragraphNumberingCheck(self, paragraphNode):
            result = False

            for node in paragraphNode.iter(f"{{{self.namespaceMain}}}numPr"):
                result = True

            return result

        def _styleBuild(self, styleRootNode):
            resultObject = {}

            for styleNode in styleRootNode.iter(f"{{{self.namespaceMain}}}style"):
                styleId = styleNode.attrib.get(f"{{{self.namespaceMain}}}styleId", "")

                if styleId != "":
                    outlineLevel = -1

                    outlineNode = styleNode.find(f"{{{self.namespaceMain}}}pPr/{{{self.namespaceMain}}}outlineLvl")

                    if outlineNode is not None and self.office._xmlNodeValue(outlineNode, self.namespaceMain) != "":
                        outlineLevel = int(self.office._xmlNodeValue(outlineNode, self.namespaceMain))

                    name = ""

                    nameNode = styleNode.find(f"{{{self.namespaceMain}}}name")

                    if nameNode is not None:
                        name = self.office._xmlNodeValue(nameNode, self.namespaceMain).lower()

                    resultObject[styleId] = {"outlineLevel": outlineLevel, "name": name}

            return resultObject

        def _paragraphDrawingCollect(self, paragraphNode):
            resultList = []

            for node in paragraphNode.iter():
                tag = self.office._xmlNodeTag(node)

                if tag == "drawing":
                    chartNode = node.find(f".//{{{self.namespaceChart}}}chart")

                    if chartNode is not None:
                        resultList.append({"kind": "image", "relationshipId": chartNode.attrib.get(f"{{{self.namespaceRelationship}}}id", ""), "isChart": True})
                    else:
                        relationshipId = ""

                        blipNode = node.find(f".//{{{self.namespaceDrawing}}}blip")

                        if blipNode is not None:
                            relationshipId = blipNode.attrib.get(f"{{{self.namespaceRelationship}}}embed", "")

                        resultList.append({"kind": "image", "relationshipId": relationshipId, "isChart": False})
                elif tag == "pict":
                    resultList.append({"kind": "image", "relationshipId": "", "isChart": False})

            return resultList

        def _blockParagraph(self, paragraphNode, styleObject, isWrapped, sizeDocument):
            resultList = []

            text = self._paragraphText(paragraphNode)

            if len(text) > 0:
                style = self._paragraphStyle(paragraphNode)
                styleDetailObject = styleObject.get(style, {"outlineLevel": -1, "name": ""})

                outlineLevel = self._paragraphOutlineLevel(paragraphNode)

                if outlineLevel == -1:
                    outlineLevel = styleDetailObject["outlineLevel"]

                resultList.append({
                    "kind": "paragraph",
                    "text": text,
                    "size": self._paragraphSize(sizeDocument, paragraphNode),
                    "style": style,
                    "styleName": styleDetailObject["name"],
                    "outlineLevel": outlineLevel,
                    "isList": self._paragraphNumberingCheck(paragraphNode),
                    "isAside": isWrapped and len(text) <= self.levelAsideLength,
                    "isContinuation": False,
                    "isWrapped": isWrapped
                })

            drawingList = self._paragraphDrawingCollect(paragraphNode)

            for a in range(len(drawingList)):
                resultList.append(drawingList[a])

            for node in paragraphNode.iter():
                if self.office._xmlNodeTag(node) == "txbxContent":
                    textboxList = self._blockWrapped(node, styleObject, sizeDocument)

                    for a in range(len(textboxList)):
                        resultList.append(textboxList[a])

            return resultList

        def _blockWrapped(self, containerNode, styleObject, sizeDocument):
            blockList = []

            for node in containerNode:
                tag = self.office._xmlNodeTag(node)

                childList = []

                if tag == "p":
                    childList = self._blockParagraph(node, styleObject, True, sizeDocument)
                elif tag == "tbl":
                    childList = self._blockTable(node, styleObject, sizeDocument)

                for a in range(len(childList)):
                    blockList.append(childList[a])

            resultList = []

            for a in range(len(blockList)):
                block = blockList[a]

                isMerge = False

                if block["kind"] == "paragraph" and block["isWrapped"] and len(resultList) > 0:
                    previous = resultList[len(resultList) - 1]

                    if previous["kind"] == "paragraph" and previous["isWrapped"]:
                        isMerge = True

                        separator = "" if self._wideCheck(previous["text"][-1:]) and self._wideCheck(block["text"][0:1]) else " "

                        previous["text"] = f"{previous['text']}{separator}{block['text']}"

                if isMerge == False:
                    resultList.append(block)

            for a in range(len(resultList)):
                if resultList[a]["kind"] == "paragraph" and resultList[a]["isWrapped"]:
                    resultList[a]["isAside"] = len(resultList[a]["text"]) <= self.levelAsideLength

            return resultList

        def _cellPropertyValue(self, cellNode, tagProperty):
            result = None

            for node in cellNode:
                if self.office._xmlNodeTag(node) == "tcPr":
                    for nodeProperty in node:
                        if self.office._xmlNodeTag(nodeProperty) == tagProperty:
                            result = self.office._xmlNodeValue(nodeProperty, self.namespaceMain)

            return result

        def _cellColumnSpan(self, cellNode):
            value = self._cellPropertyValue(cellNode, "gridSpan")

            if value is None or value == "":
                return 1

            return int(value)

        def _cellMergeVertical(self, cellNode):
            value = self._cellPropertyValue(cellNode, "vMerge")

            if value is None:
                return ""

            return value if value != "" else "continue"

        def _blockTable(self, tableNode, styleObject, sizeDocument):
            resultList = []

            rowNodeList = []

            for node in tableNode:
                if self.office._xmlNodeTag(node) == "tr":
                    rowNodeList.append(node)

            columnMax = 0
            isData = len(rowNodeList) >= 2

            rowCellList = []
            openObject = {}

            for a in range(len(rowNodeList)):
                cellNodeList = []

                for node in rowNodeList[a]:
                    if self.office._xmlNodeTag(node) == "tc":
                        cellNodeList.append(node)

                cellList = []
                columnIndex = 0

                for b in range(len(cellNodeList)):
                    textList = []

                    for node in cellNodeList[b]:
                        tag = self.office._xmlNodeTag(node)

                        if tag == "p":
                            text = self._paragraphText(node)

                            if len(text) > 0:
                                textList.append(text)

                                if len(text) > self.levelAsideLength:
                                    isData = False
                        elif tag == "tbl":
                            isData = False

                    if len(self._paragraphDrawingCollect(cellNodeList[b])) > 0:
                        isData = False

                    columnSpan = self._cellColumnSpan(cellNodeList[b])

                    if self._cellMergeVertical(cellNodeList[b]) == "continue" and columnIndex in openObject:
                        openObject[columnIndex]["rowSpan"] += 1
                    else:
                        cellObject = {"text": " ".join(textList), "rowSpan": 1, "columnSpan": columnSpan}

                        cellList.append(cellObject)

                        openObject[columnIndex] = cellObject

                    columnIndex += columnSpan

                columnMax = max(columnMax, columnIndex)

                rowCellList.append(cellList)

            if columnMax < 2:
                isData = False

            if isData:
                for a in range(len(rowCellList)):
                    resultList.append({"kind": "tableRow", "cellList": rowCellList[a]})
            else:
                for a in range(len(rowNodeList)):
                    for node in rowNodeList[a]:
                        if self.office._xmlNodeTag(node) == "tc":
                            blockList = self._blockWrapped(node, styleObject, sizeDocument)

                            for b in range(len(blockList)):
                                resultList.append(blockList[b])

            return resultList

        def _blockBuild(self, containerNode, styleObject, sizeDocument):
            resultList = []

            for node in containerNode:
                tag = self.office._xmlNodeTag(node)

                blockList = []

                if tag == "p":
                    blockList = self._blockParagraph(node, styleObject, False, sizeDocument)
                elif tag == "tbl":
                    blockList = self._blockTable(node, styleObject, sizeDocument)
                elif tag != "sectPr":
                    blockList = self._blockBuild(node, styleObject, sizeDocument)

                for a in range(len(blockList)):
                    resultList.append(blockList[a])

            return resultList

        def _asideRunFlush(self, runIndexList, blockList):
            if len(runIndexList) >= self.levelAsideCount:
                for a in range(len(runIndexList)):
                    blockList[runIndexList[a]]["isAside"] = True

        def _wideCheck(self, character):
            return character != "" and unicodedata.east_asian_width(character) in ("W", "F")

        def _bodyTextCheck(self, block):
            result = False

            if block["kind"] == "paragraph" and block["outlineLevel"] == -1 and block["isList"] == False:
                if block["style"] != "Title" and block["styleName"] != "title" and re.match(r"Heading(\d)", block["style"]) is None:
                    if "caption" not in block["styleName"] and "Caption" not in block["style"] and "didascalia" not in block["styleName"]:
                        result = True

            return result

        def _sentenceEndCheck(self, text):
            textClean = re.sub(r"(\s*\[[^\[\]]{1,20}\]|[)\]}\"'”’»›])+$", "", text.strip())

            return len(textClean) > 0 and textClean[-1:] in self.sentenceEndList

        def _continuationCheck(self, previous, block):
            result = False

            if previous is not None and previous["kind"] == "paragraph" and previous["isAside"] == False:
                if len(previous["text"]) > self.levelAsideLength and previous["size"] == block["size"]:
                    if self._sentenceEndCheck(previous["text"]) == False:
                        result = True

            return result

        def _previousParagraph(self, index, blockList):
            result = None

            for a in range(index - 1, -1, -1):
                block = blockList[a]

                if block["kind"] == "paragraph":
                    if block["isAside"] == False:
                        result = block

                        break
                elif block["kind"] != "image":
                    break

            return result

        def _previousBlock(self, index, blockList):
            result = None

            for a in range(index - 1, -1, -1):
                block = blockList[a]

                if block["kind"] == "image":
                    continue

                if block["kind"] == "paragraph":
                    result = block

                break

            return result

        def _continuationMark(self, blockList):
            for a in range(len(blockList)):
                block = blockList[a]

                if block["kind"] == "paragraph" and block["isAside"] == False and block["isContinuation"] == False and self._bodyTextCheck(block):
                    isChained = False

                    if block["isWrapped"]:
                        previousBlock = self._previousBlock(a, blockList)

                        if previousBlock is not None and previousBlock["isAside"] and previousBlock["size"] == block["size"]:
                            if self._sentenceEndCheck(previousBlock["text"]) == False:
                                block["isAside"] = True

                                isChained = True

                    if isChained == False and (len(block["text"]) <= self.levelAsideLength or block["text"][0:1].islower()):
                        previous = self._previousParagraph(a, blockList)

                        if self._continuationCheck(previous, block):
                            block["isContinuation"] = True

            return blockList

        def _asideMark(self, blockList):
            runIndexList = []

            for a in range(len(blockList)):
                block = blockList[a]

                isBreak = True

                if block["kind"] == "image":
                    isBreak = False
                elif block["kind"] == "paragraph":
                    if block["isAside"]:
                        isBreak = False
                    elif self._bodyTextCheck(block) and len(block["text"]) <= self.levelAsideLength:
                        if len(runIndexList) == 0 and self._continuationCheck(blockList[a - 1] if a > 0 else None, block):
                            block["isContinuation"] = True
                        else:
                            runIndexList.append(a)

                            isBreak = False

                if isBreak:
                    self._asideRunFlush(runIndexList, blockList)

                    runIndexList = []

            self._asideRunFlush(runIndexList, blockList)

            return blockList

        def _bodySize(self, blockList):
            countObject = {}

            for a in range(len(blockList)):
                if blockList[a]["kind"] == "paragraph" and blockList[a]["isAside"] == False:
                    size = blockList[a]["size"]

                    countObject[size] = countObject.get(size, 0) + len(blockList[a]["text"])

            result = 0.0
            countMax = 0

            for size in countObject:
                if countObject[size] > countMax:
                    countMax = countObject[size]
                    result = size

            return result

        def _titleSizeRank(self, blockList, bodySize):
            resultList = []

            for a in range(len(blockList)):
                block = blockList[a]

                if self._bodyTextCheck(block) and block["isAside"] == False:
                    if bodySize > 0 and block["size"] >= bodySize * self.levelTitleSize and len(block["text"]) <= self.levelTitleLength:
                        if block["size"] not in resultList:
                            resultList.append(block["size"])

            resultList.sort(reverse=True)

            return resultList

        def _itemLabel(self, block, titleSizeList, isDocTitleFound):
            resultObject = {"label": "text", "level": 0}

            styleMatch = re.match(r"Heading(\d)", block["style"])

            if block["style"] == "Title" or block["styleName"] == "title":
                resultObject["label"] = "doc_title"
            elif "Caption" in block["style"] or "caption" in block["styleName"] or "didascalia" in block["styleName"]:
                resultObject["label"] = "figure_title"
            elif styleMatch is not None:
                resultObject["label"] = "paragraph_title"
                resultObject["level"] = int(styleMatch.group(1)) + 1
            elif block["outlineLevel"] >= 0:
                resultObject["label"] = "paragraph_title"
                resultObject["level"] = block["outlineLevel"] + 2
            elif block["size"] in titleSizeList and len(block["text"]) <= self.levelTitleLength:
                index = titleSizeList.index(block["size"])

                if index == 0 and isDocTitleFound == False:
                    resultObject["label"] = "doc_title"
                else:
                    resultObject["label"] = "paragraph_title"
                    resultObject["level"] = max(2, 1 + index) if isDocTitleFound else 2 + index

            return resultObject

        def execute(self, pathInput, pathOutput):
            zipFile = zipfile.ZipFile(pathInput)

            rootNode = self.office._xmlRootBuild("word/document.xml", zipFile)

            sizeDocument = 22.0
            styleObject = {}

            styleRootNode = self.office._xmlRootBuild("word/styles.xml", zipFile)

            if styleRootNode is not None:
                styleObject = self._styleBuild(styleRootNode)

                sizeNode = styleRootNode.find(
                    f"{{{self.namespaceMain}}}docDefaults/{{{self.namespaceMain}}}rPrDefault/{{{self.namespaceMain}}}rPr/{{{self.namespaceMain}}}sz"
                )

                if sizeNode is not None and self.office._xmlNodeValue(sizeNode, self.namespaceMain) != "":
                    sizeDocument = float(self.office._xmlNodeValue(sizeNode, self.namespaceMain))

            relationshipRootNode = self.office._xmlRootBuild("word/_rels/document.xml.rels", zipFile)

            pathObject = {}

            if relationshipRootNode is not None:
                for node in relationshipRootNode.iter(f"{{{self.namespacePackage}}}Relationship"):
                    target = node.attrib.get("Target", "")

                    pathObject[node.attrib.get("Id", "")] = target[1:] if target.startswith("/") else f"word/{target}"

            blockList = []

            if rootNode is not None:
                self._fallbackRemove(rootNode)

                bodyNode = rootNode.find(f"{{{self.namespaceMain}}}body")

                if bodyNode is not None:
                    blockList = self._blockBuild(bodyNode, styleObject, sizeDocument)

            blockList = self._asideMark(blockList)
            blockList = self._continuationMark(blockList)

            bodySize = self._bodySize(blockList)
            titleSizeList = self._titleSizeRank(blockList, bodySize)

            itemMainList = []
            itemSecondaryList = []

            isDocTitleFound = False

            for a in range(len(blockList)):
                block = blockList[a]

                if block["kind"] == "tableRow":
                    textList = []

                    for b in range(len(block["cellList"])):
                        textList.append(block["cellList"][b]["text"])

                    itemMainList.append({"label": "tableRow", "text": " | ".join(textList), "cellList": block["cellList"]})
                elif block["kind"] == "image":
                    item = {"label": "image", "text": ""}

                    pathTarget = pathObject[block["relationshipId"]] if block["relationshipId"] in pathObject else ""

                    if block["isChart"]:
                        item["label"] = "chart"

                        chartRootNode = self.office._xmlRootBuild(pathTarget, zipFile)

                        if chartRootNode is not None:
                            item["text"] = self.office._xmlChartText(chartRootNode)
                    elif pathTarget in zipFile.namelist():
                        os.makedirs(f"{pathOutput}media/", exist_ok=True)

                        with open(f"{pathOutput}media/{os.path.basename(pathTarget)}", "wb") as file:
                            file.write(zipFile.read(pathTarget))

                        item["path"] = f"media/{os.path.basename(pathTarget)}"

                    itemSecondaryList.append(item)
                elif block["isAside"]:
                    itemSecondaryList.append({"label": "aside_text", "text": block["text"]})
                elif block["isContinuation"] and len(itemMainList) > 0:
                    itemPrevious = itemMainList[len(itemMainList) - 1]

                    if self._wideCheck(itemPrevious["text"][-1:]) and self._wideCheck(block["text"][0:1]):
                        itemPrevious["text"] += block["text"]
                    else:
                        itemPrevious["text"] += f" {block['text']}"
                else:
                    labelObject = self._itemLabel(block, titleSizeList, isDocTitleFound)

                    if labelObject["label"] == "doc_title":
                        isDocTitleFound = True

                    item = {"label": labelObject["label"], "text": block["text"]}

                    if labelObject["label"] == "paragraph_title":
                        item["level"] = labelObject["level"]

                    if labelObject["label"] == "text" and block["isList"]:
                        item["isList"] = True

                    if labelObject["label"] == "figure_title":
                        itemSecondaryList.append(item)
                    else:
                        itemMainList.append(item)

            zipFile.close()

            for a in range(len(itemMainList)):
                itemMainList[a]["flow"] = "main"
                itemMainList[a]["order"] = a + 1

            for a in range(len(itemSecondaryList)):
                itemSecondaryList[a]["flow"] = "secondary"
                itemSecondaryList[a]["order"] = a + 1

            resultObject = {
                "pageList": [
                    {"number": 1, "itemMainList": itemMainList, "itemSecondaryList": itemSecondaryList}
                ]
            }

            with open(f"{pathOutput}debug/layout/ast.json", "w", encoding="utf-8") as file:
                json.dump(resultObject, file, ensure_ascii=False, indent=4)

            return resultObject

        def __init__(self, office):
            self.office = office

            self.namespaceMain = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            self.namespaceDrawing = "http://schemas.openxmlformats.org/drawingml/2006/main"
            self.namespaceChart = "http://schemas.openxmlformats.org/drawingml/2006/chart"
            self.namespaceRelationship = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
            self.namespacePackage = "http://schemas.openxmlformats.org/package/2006/relationships"

            self.levelTitleSize = 1.15
            self.levelTitleLength = 120
            self.levelAsideLength = 60
            self.levelAsideCount = 4

            self.sentenceEndList = [".", "!", "?", "…", ";", ":", "。", "！", "？", "；", "："]

    class Xlsx:
        def _cellColumn(self, reference):
            result = 0

            for a in range(len(reference)):
                if reference[a].isalpha():
                    result = result * 26 + (ord(reference[a].upper()) - 64)
                else:
                    break

            return max(0, result - 1)

        def _stringText(self, node):
            result = ""

            tag = self.office._xmlNodeTag(node)

            if tag != "rPh":
                if tag == "t":
                    result += node.text if node.text is not None else ""

                for childNode in node:
                    result += self._stringText(childNode)

            return result

        def _sharedStringBuild(self, sharedStringRootNode):
            resultList = []

            for node in sharedStringRootNode.iter(f"{{{self.namespaceMain}}}si"):
                resultList.append(self._stringText(node))

            return resultList

        def _dateFormatCheck(self, formatCode):
            formatClean = re.sub(r"\"[^\"]*\"|\[[^\]]*\]|\\.", "", formatCode)

            return re.search(r"[dmyhs]", formatClean, re.IGNORECASE) is not None

        def _dateStyleBuild(self, styleRootNode):
            resultList = []

            numberFormatObject = {}

            for node in styleRootNode.iter(f"{{{self.namespaceMain}}}numFmt"):
                numberFormatObject[int(node.attrib.get("numFmtId", "-1"))] = node.attrib.get("formatCode", "")

            cellFormatNode = styleRootNode.find(f"{{{self.namespaceMain}}}cellXfs")

            if cellFormatNode is not None:
                indexFormat = 0

                for node in cellFormatNode:
                    if self.office._xmlNodeTag(node) == "xf":
                        numberFormatId = int(node.attrib.get("numFmtId", "0"))

                        isDate = numberFormatId in self.numberFormatDateList

                        if isDate == False and numberFormatId in numberFormatObject:
                            isDate = self._dateFormatCheck(numberFormatObject[numberFormatId])

                        if isDate:
                            resultList.append(indexFormat)

                        indexFormat += 1

            return resultList

        def _numberCheck(self, text):
            return re.match(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?$", text) is not None

        def _numberText(self, text):
            result = text

            value = float(text)

            if value == int(value):
                result = str(int(value))

            return result

        def _dateText(self, value):
            dateValue = datetime.datetime(1899, 12, 30) + datetime.timedelta(days=value)

            result = dateValue.strftime("%Y-%m-%d %H:%M:%S")

            if value < 1.0:
                result = dateValue.strftime("%H:%M:%S")
            elif value == int(value):
                result = dateValue.strftime("%Y-%m-%d")

            return result

        def _cellText(self, cellNode, sharedStringList, dateStyleList):
            result = ""

            cellType = cellNode.attrib.get("t", "n")

            valueNode = cellNode.find(f"{{{self.namespaceMain}}}v")
            valueText = valueNode.text if valueNode is not None and valueNode.text is not None else ""

            if cellType == "s":
                if valueText != "" and int(valueText) < len(sharedStringList):
                    result = sharedStringList[int(valueText)]
            elif cellType == "inlineStr":
                inlineNode = cellNode.find(f"{{{self.namespaceMain}}}is")

                if inlineNode is not None:
                    result = self._stringText(inlineNode)
            elif cellType == "b":
                result = "TRUE" if valueText == "1" else "FALSE"
            elif cellType == "str" or cellType == "e":
                result = valueText
            else:
                result = valueText

                if valueText != "" and self._numberCheck(valueText):
                    styleText = cellNode.attrib.get("s", "")
                    styleIndex = int(styleText) if styleText.isdigit() else -1

                    if styleIndex in dateStyleList:
                        result = self._dateText(float(valueText))
                    else:
                        result = self._numberText(valueText)

            return result.strip()

        def _pivotRangeCollect(self, zipFile, sheetPath):
            resultList = []

            relationshipRootNode = self.office._xmlRootBuild(f"{os.path.dirname(sheetPath)}/_rels/{os.path.basename(sheetPath)}.rels", zipFile)

            if relationshipRootNode is not None:
                for node in relationshipRootNode.iter(f"{{{self.namespacePackage}}}Relationship"):
                    if node.attrib.get("Type", "").endswith("/pivotTable"):
                        target = node.attrib.get("Target", "")
                        pathPivot = target[1:] if target.startswith("/") else os.path.normpath(f"{os.path.dirname(sheetPath)}/{target}")

                        pivotRootNode = self.office._xmlRootBuild(pathPivot, zipFile)

                        if pivotRootNode is not None:
                            locationNode = pivotRootNode.find(f"{{{self.namespaceMain}}}location")

                            if locationNode is not None:
                                referenceSplit = locationNode.attrib.get("ref", "").split(":")

                                if len(referenceSplit) == 2:
                                    resultList.append({
                                        "rowFirst": int(re.sub(r"[A-Za-z]", "", referenceSplit[0])),
                                        "rowLast": int(re.sub(r"[A-Za-z]", "", referenceSplit[1])),
                                        "columnFirst": self._cellColumn(referenceSplit[0]),
                                        "columnLast": self._cellColumn(referenceSplit[1])
                                    })

            return resultList

        def _drawingCollect(self, zipFile, sheetPath, pathOutput):
            resultList = []

            relationshipRootNode = self.office._xmlRootBuild(f"{os.path.dirname(sheetPath)}/_rels/{os.path.basename(sheetPath)}.rels", zipFile)

            if relationshipRootNode is not None:
                for node in relationshipRootNode.iter(f"{{{self.namespacePackage}}}Relationship"):
                    if node.attrib.get("Type", "").endswith("/drawing"):
                        target = node.attrib.get("Target", "")
                        pathDrawing = target[1:] if target.startswith("/") else os.path.normpath(f"{os.path.dirname(sheetPath)}/{target}")

                        drawingRootNode = self.office._xmlRootBuild(pathDrawing, zipFile)
                        drawingRelationshipRootNode = self.office._xmlRootBuild(f"{os.path.dirname(pathDrawing)}/_rels/{os.path.basename(pathDrawing)}.rels", zipFile)

                        pathObject = {}

                        if drawingRelationshipRootNode is not None:
                            for relationshipNode in drawingRelationshipRootNode.iter(f"{{{self.namespacePackage}}}Relationship"):
                                targetDrawing = relationshipNode.attrib.get("Target", "")

                                pathObject[relationshipNode.attrib.get("Id", "")] = targetDrawing[1:] if targetDrawing.startswith("/") else os.path.normpath(f"{os.path.dirname(pathDrawing)}/{targetDrawing}")

                        if drawingRootNode is not None:
                            for drawingNode in drawingRootNode.iter():
                                tag = self.office._xmlNodeTag(drawingNode)

                                if tag == "graphicFrame":
                                    item = {"label": "image", "text": ""}

                                    chartNode = drawingNode.find(f".//{{{self.namespaceChart}}}chart")

                                    if chartNode is not None:
                                        item["label"] = "chart"

                                        relationshipId = chartNode.attrib.get(f"{{{self.namespaceRelationship}}}id", "")
                                        pathChart = pathObject[relationshipId] if relationshipId in pathObject else ""

                                        chartRootNode = self.office._xmlRootBuild(pathChart, zipFile)

                                        if chartRootNode is not None:
                                            item["text"] = self.office._xmlChartText(chartRootNode)

                                    resultList.append(item)
                                elif tag == "pic":
                                    item = {"label": "image", "text": ""}

                                    blipNode = drawingNode.find(f".//{{{self.namespaceDrawing}}}blip")

                                    if blipNode is not None:
                                        relationshipId = blipNode.attrib.get(f"{{{self.namespaceRelationship}}}embed", "")
                                        pathMedia = pathObject[relationshipId] if relationshipId in pathObject else ""

                                        if pathMedia in zipFile.namelist():
                                            os.makedirs(f"{pathOutput}media/", exist_ok=True)

                                            with open(f"{pathOutput}media/{os.path.basename(pathMedia)}", "wb") as file:
                                                file.write(zipFile.read(pathMedia))

                                            item["path"] = f"media/{os.path.basename(pathMedia)}"

                                    resultList.append(item)

            return resultList

        def _rowCollect(self, sheetRootNode, sharedStringList, dateStyleList, pivotRangeList):
            rowObjectList = []

            rowNumberNext = 1

            for rowNode in sheetRootNode.iter(f"{{{self.namespaceMain}}}row"):
                rowNumberText = rowNode.attrib.get("r", "")
                rowNumber = int(rowNumberText) if rowNumberText.isdigit() else rowNumberNext

                cellList = []
                columnNext = 0

                for cellNode in rowNode:
                    if self.office._xmlNodeTag(cellNode) == "c":
                        reference = cellNode.attrib.get("r", "")
                        column = self._cellColumn(reference) if reference != "" else columnNext

                        while len(cellList) < column:
                            cellList.append("")

                        cellText = self._cellText(cellNode, sharedStringList, dateStyleList)

                        for a in range(len(pivotRangeList)):
                            if pivotRangeList[a]["rowFirst"] <= rowNumber <= pivotRangeList[a]["rowLast"] and pivotRangeList[a]["columnFirst"] <= column <= pivotRangeList[a]["columnLast"]:
                                cellText = ""

                                break

                        cellList.append(cellText)

                        columnNext = column + 1

                while len(cellList) > 0 and cellList[len(cellList) - 1] == "":
                    cellList.pop()

                rowObjectList.append({"number": rowNumber, "cellList": cellList})

                rowNumberNext = rowNumber + 1

            numberFirst = 0
            numberLast = 0
            columnCount = 0

            for a in range(len(rowObjectList)):
                if len(rowObjectList[a]["cellList"]) > 0:
                    if numberFirst == 0:
                        numberFirst = rowObjectList[a]["number"]

                    numberLast = rowObjectList[a]["number"]
                    columnCount = max(columnCount, len(rowObjectList[a]["cellList"]))

            cellListObject = {}

            for a in range(len(rowObjectList)):
                cellListObject[rowObjectList[a]["number"]] = rowObjectList[a]["cellList"]

            resultList = []

            if numberFirst > 0:
                for a in range(numberFirst, numberLast + 1):
                    cellList = cellListObject[a] if a in cellListObject else []

                    while len(cellList) < columnCount:
                        cellList.append("")

                    resultList.append({"number": a, "cellList": cellList})

            return resultList

        def _mergeCollect(self, sheetRootNode):
            resultList = []

            for node in sheetRootNode.iter(f"{{{self.namespaceMain}}}mergeCell"):
                reference = node.attrib.get("ref", "")

                if reference != "":
                    resultList.append(reference)

            return resultList

        def _sheetBuild(self, zipFile):
            resultList = []

            workbookRootNode = self.office._xmlRootBuild("xl/workbook.xml", zipFile)
            relationshipRootNode = self.office._xmlRootBuild("xl/_rels/workbook.xml.rels", zipFile)

            pathObject = {}

            if relationshipRootNode is not None:
                for node in relationshipRootNode.iter(f"{{{self.namespacePackage}}}Relationship"):
                    target = node.attrib.get("Target", "")

                    pathObject[node.attrib.get("Id", "")] = target[1:] if target.startswith("/") else f"xl/{target}"

            if workbookRootNode is not None:
                for node in workbookRootNode.iter(f"{{{self.namespaceMain}}}sheet"):
                    relationshipId = node.attrib.get(f"{{{self.namespaceRelationship}}}id", "")

                    if relationshipId in pathObject and "worksheets/" in pathObject[relationshipId]:
                        resultList.append({"name": node.attrib.get("name", ""), "path": pathObject[relationshipId]})

            return resultList

        def execute(self, pathInput, pathOutput):
            zipFile = zipfile.ZipFile(pathInput)

            sharedStringList = []
            dateStyleList = []

            sharedStringRootNode = self.office._xmlRootBuild("xl/sharedStrings.xml", zipFile)

            if sharedStringRootNode is not None:
                sharedStringList = self._sharedStringBuild(sharedStringRootNode)

            styleRootNode = self.office._xmlRootBuild("xl/styles.xml", zipFile)

            if styleRootNode is not None:
                dateStyleList = self._dateStyleBuild(styleRootNode)

            sheetList = self._sheetBuild(zipFile)

            pageList = []
            rowCount = 0

            for a in range(len(sheetList)):
                sheetRootNode = self.office._xmlRootBuild(sheetList[a]["path"], zipFile)

                pivotRangeList = self._pivotRangeCollect(zipFile, sheetList[a]["path"])

                rowList = self._rowCollect(sheetRootNode, sharedStringList, dateStyleList, pivotRangeList) if sheetRootNode is not None else []
                mergeList = self._mergeCollect(sheetRootNode) if sheetRootNode is not None else []

                itemMainList = [{"label": "sheetName", "text": sheetList[a]["name"]}]

                for b in range(len(rowList)):
                    itemMainList.append({"label": "tableRow", "number": rowList[b]["number"], "text": " | ".join(rowList[b]["cellList"]), "cellList": rowList[b]["cellList"]})

                for b in range(len(itemMainList)):
                    itemMainList[b]["flow"] = "main"
                    itemMainList[b]["order"] = b + 1

                itemSecondaryList = self._drawingCollect(zipFile, sheetList[a]["path"], pathOutput)

                for b in range(len(itemSecondaryList)):
                    itemSecondaryList[b]["flow"] = "secondary"
                    itemSecondaryList[b]["order"] = b + 1

                pageList.append({"number": a + 1, "mergeList": mergeList, "itemMainList": itemMainList, "itemSecondaryList": itemSecondaryList})

                rowCount += len(rowList)

            zipFile.close()

            resultObject = {"pageList": pageList}

            with open(f"{pathOutput}debug/layout/ast.json", "w", encoding="utf-8") as file:
                json.dump(resultObject, file, ensure_ascii=False, indent=4)

            return resultObject

        def __init__(self, office):
            self.office = office

            self.namespaceMain = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
            self.namespaceDrawing = "http://schemas.openxmlformats.org/drawingml/2006/main"
            self.namespaceChart = "http://schemas.openxmlformats.org/drawingml/2006/chart"
            self.namespaceRelationship = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
            self.namespacePackage = "http://schemas.openxmlformats.org/package/2006/relationships"

            self.numberFormatDateList = [14, 15, 16, 17, 18, 19, 20, 21, 22, 45, 46, 47]

    class Pptx:
        def _textCollect(self, node):
            result = ""

            tag = self.office._xmlNodeTag(node)

            if tag == "t":
                result += node.text if node.text is not None else ""
            elif tag == "br" or tag == "tab":
                result += " "

            for childNode in node:
                result += self._textCollect(childNode)

            return result

        def _paragraphText(self, paragraphNode):
            return self._textCollect(paragraphNode).strip()

        def _placeholderType(self, shapeNode):
            result = ""

            placeholderNode = shapeNode.find(f"{{{self.namespaceMain}}}nvSpPr/{{{self.namespaceMain}}}nvPr/{{{self.namespaceMain}}}ph")

            if placeholderNode is not None:
                result = placeholderNode.attrib.get("type", "body")

            return result

        def _blockTable(self, tableNode):
            resultList = []

            for rowNode in tableNode:
                if self.office._xmlNodeTag(rowNode) == "tr":
                    cellList = []

                    for cellNode in rowNode:
                        if self.office._xmlNodeTag(cellNode) == "tc":
                            if cellNode.attrib.get("hMerge", "") == "1" or cellNode.attrib.get("vMerge", "") == "1":
                                continue

                            textList = []

                            for paragraphNode in cellNode.iter(f"{{{self.namespaceDrawing}}}p"):
                                text = self._paragraphText(paragraphNode)

                                if len(text) > 0:
                                    textList.append(text)

                            cellList.append({
                                "text": " ".join(textList),
                                "rowSpan": int(cellNode.attrib.get("rowSpan", "1")),
                                "columnSpan": int(cellNode.attrib.get("gridSpan", "1"))
                            })

                    resultList.append({"kind": "tableRow", "cellList": cellList})

            return resultList

        def _blockBuild(self, containerNode):
            resultList = []

            for node in containerNode:
                tag = self.office._xmlNodeTag(node)

                if tag == "sp":
                    placeholderType = self._placeholderType(node)

                    if placeholderType not in self.placeholderSkipList:
                        textBodyNode = node.find(f"{{{self.namespaceMain}}}txBody")

                        if textBodyNode is not None:
                            for paragraphNode in textBodyNode:
                                if self.office._xmlNodeTag(paragraphNode) == "p":
                                    text = self._paragraphText(paragraphNode)

                                    if len(text) > 0:
                                        resultList.append({"kind": "paragraph", "text": text, "placeholderType": placeholderType})
                elif tag == "graphicFrame":
                    chartNode = node.find(f".//{{{self.namespaceChart}}}chart")
                    tableNode = node.find(f".//{{{self.namespaceDrawing}}}tbl")

                    if chartNode is not None:
                        resultList.append({"kind": "chart", "relationshipId": chartNode.attrib.get(f"{{{self.namespaceRelationship}}}id", "")})
                    elif tableNode is not None:
                        blockList = self._blockTable(tableNode)

                        for a in range(len(blockList)):
                            resultList.append(blockList[a])
                    else:
                        resultList.append({"kind": "image", "relationshipId": ""})
                elif tag == "pic":
                    relationshipId = ""

                    blipNode = node.find(f".//{{{self.namespaceDrawing}}}blip")

                    if blipNode is not None:
                        relationshipId = blipNode.attrib.get(f"{{{self.namespaceRelationship}}}embed", "")

                    resultList.append({"kind": "image", "relationshipId": relationshipId})
                elif tag == "grpSp":
                    blockList = self._blockBuild(node)

                    for a in range(len(blockList)):
                        resultList.append(blockList[a])

            return resultList

        def _relationshipBuild(self, zipFile, pathFile):
            resultObject = {}

            relationshipRootNode = self.office._xmlRootBuild(f"{os.path.dirname(pathFile)}/_rels/{os.path.basename(pathFile)}.rels", zipFile)

            if relationshipRootNode is not None:
                for node in relationshipRootNode.iter(f"{{{self.namespacePackage}}}Relationship"):
                    target = node.attrib.get("Target", "")

                    resultObject[node.attrib.get("Id", "")] = {
                        "path": target[1:] if target.startswith("/") else os.path.normpath(f"{os.path.dirname(pathFile)}/{target}"),
                        "type": node.attrib.get("Type", "")
                    }

            return resultObject

        def _slideBuild(self, zipFile):
            resultList = []

            presentationRootNode = self.office._xmlRootBuild("ppt/presentation.xml", zipFile)

            pathObject = self._relationshipBuild(zipFile, "ppt/presentation.xml")

            if presentationRootNode is not None:
                for node in presentationRootNode.iter(f"{{{self.namespaceMain}}}sldId"):
                    relationshipId = node.attrib.get(f"{{{self.namespaceRelationship}}}id", "")

                    if relationshipId in pathObject:
                        resultList.append(pathObject[relationshipId]["path"])

            return resultList

        def _notesText(self, pathObject, zipFile):
            result = ""

            for relationshipId in pathObject:
                if pathObject[relationshipId]["type"].endswith("/notesSlide"):
                    notesRootNode = self.office._xmlRootBuild(pathObject[relationshipId]["path"], zipFile)

                    if notesRootNode is not None:
                        textList = []

                        for shapeNode in notesRootNode.iter(f"{{{self.namespaceMain}}}sp"):
                            if self._placeholderType(shapeNode) == "body":
                                for paragraphNode in shapeNode.iter(f"{{{self.namespaceDrawing}}}p"):
                                    text = self._paragraphText(paragraphNode)

                                    if len(text) > 0:
                                        textList.append(text)

                        result = "\n".join(textList)

            return result

        def execute(self, pathInput, pathOutput):
            zipFile = zipfile.ZipFile(pathInput)

            slidePathList = self._slideBuild(zipFile)

            pageList = []
            blockCount = 0

            isDocTitleFound = False

            for a in range(len(slidePathList)):
                slideRootNode = self.office._xmlRootBuild(slidePathList[a], zipFile)

                pathObject = self._relationshipBuild(zipFile, slidePathList[a])

                blockList = []

                if slideRootNode is not None:
                    treeNode = slideRootNode.find(f"{{{self.namespaceMain}}}cSld/{{{self.namespaceMain}}}spTree")

                    if treeNode is not None:
                        blockList = self._blockBuild(treeNode)

                itemMainList = []
                itemSecondaryList = []

                for b in range(len(blockList)):
                    block = blockList[b]

                    if block["kind"] == "paragraph":
                        if block["placeholderType"] == "title" or block["placeholderType"] == "ctrTitle":
                            if isDocTitleFound == False:
                                isDocTitleFound = True

                                itemMainList.append({"label": "doc_title", "text": block["text"]})
                            else:
                                itemMainList.append({"label": "paragraph_title", "level": 2, "text": block["text"]})
                        else:
                            itemMainList.append({"label": "text", "text": block["text"]})
                    elif block["kind"] == "tableRow":
                        textList = []

                        for c in range(len(block["cellList"])):
                            textList.append(block["cellList"][c]["text"])

                        itemMainList.append({"label": "tableRow", "text": " | ".join(textList), "cellList": block["cellList"]})
                    elif block["kind"] == "chart":
                        item = {"label": "chart", "text": ""}

                        pathChart = pathObject[block["relationshipId"]]["path"] if block["relationshipId"] in pathObject else ""

                        chartRootNode = self.office._xmlRootBuild(pathChart, zipFile)

                        if chartRootNode is not None:
                            item["text"] = self.office._xmlChartText(chartRootNode)

                        itemSecondaryList.append(item)
                    elif block["kind"] == "image":
                        item = {"label": "image", "text": ""}

                        pathMedia = pathObject[block["relationshipId"]]["path"] if block["relationshipId"] in pathObject else ""

                        if pathMedia in zipFile.namelist():
                            os.makedirs(f"{pathOutput}media/", exist_ok=True)

                            with open(f"{pathOutput}media/{os.path.basename(pathMedia)}", "wb") as file:
                                file.write(zipFile.read(pathMedia))

                            item["path"] = f"media/{os.path.basename(pathMedia)}"

                        itemSecondaryList.append(item)

                notesText = self._notesText(pathObject, zipFile)

                if len(notesText) > 0:
                    itemSecondaryList.append({"label": "aside_text", "text": notesText})

                for b in range(len(itemMainList)):
                    itemMainList[b]["flow"] = "main"
                    itemMainList[b]["order"] = b + 1

                for b in range(len(itemSecondaryList)):
                    itemSecondaryList[b]["flow"] = "secondary"
                    itemSecondaryList[b]["order"] = b + 1

                pageList.append({"number": a + 1, "itemMainList": itemMainList, "itemSecondaryList": itemSecondaryList})

                blockCount += len(blockList)

            zipFile.close()

            resultObject = {"pageList": pageList}

            with open(f"{pathOutput}debug/layout/ast.json", "w", encoding="utf-8") as file:
                json.dump(resultObject, file, ensure_ascii=False, indent=4)

            return resultObject

        def __init__(self, office):
            self.office = office

            self.namespaceMain = "http://schemas.openxmlformats.org/presentationml/2006/main"
            self.namespaceDrawing = "http://schemas.openxmlformats.org/drawingml/2006/main"
            self.namespaceChart = "http://schemas.openxmlformats.org/drawingml/2006/chart"
            self.namespaceRelationship = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
            self.namespacePackage = "http://schemas.openxmlformats.org/package/2006/relationships"

            self.placeholderSkipList = ["sldNum", "dt", "ftr"]

    def __init__(self):
        self.readerObject = {
            ".docx": Office.Docx(self),
            ".xlsx": Office.Xlsx(self),
            ".pptx": Office.Pptx(self)
        }
