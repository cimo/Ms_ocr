import sys
import os
import glob
import subprocess
import cv2
import math
import unicodedata
import zlib
import codecs
import re

sys.dont_write_bytecode = True

class Process:
    def _pageBuild(self, pathInput, pathOutput):
        subprocess.run(["pdftoppm", "-jpeg", "-scale-to", "1755", pathInput, f"{pathOutput}page/page"], capture_output=True, text=True)

        pathFileList = glob.glob(f"{pathOutput}page/page-*.jpg")

        resultList = []

        for a in range(len(pathFileList)):
            numberPage = int(os.path.splitext(os.path.basename(pathFileList[a]))[0].split("-")[1])

            pathPage = f"{pathOutput}page/{numberPage}.jpg"

            os.rename(pathFileList[a], pathPage)

            resultList.append({"number": numberPage, "image": cv2.imread(pathPage)})

        return sorted(resultList, key=lambda pageObject: pageObject["number"])

    def _itemBuild(self, image, pageReader, countStart, numberPage):
        imageHeight, imageWidth = image.shape[0:2]

        scaleX = imageWidth / pageReader["width"]
        scaleY = imageHeight / pageReader["height"]

        elementList = pageReader["elementList"]

        resultList = []

        for a in range(len(elementList)):
            if elementList[a]["type"] != "text":
                continue

            bboxList = [
                int(round(elementList[a]["x0"] * scaleX)),
                int(round(elementList[a]["y0"] * scaleY)),
                int(round(elementList[a]["x1"] * scaleX)),
                int(round(elementList[a]["y1"] * scaleY))
            ]

            resultList.append({
                "id": countStart + len(resultList) + 1,
                "page": numberPage,
                "bbox": bboxList,
                "centerPoint": self._centerPointCalculate(bboxList),
                "text": elementList[a]["text"],
                "isMatch": False
            })

        return resultList

    def _centerPointCalculate(self, bboxList):
        return {
            "x": int(round((bboxList[0] + bboxList[2]) / 2)),
            "y": int(round((bboxList[1] + bboxList[3]) / 2))
        }

    def _debugText(self, image, itemList, pathOutput, numberPage):
        imageDebug = image.copy()

        for a in range(len(itemList)):
            bboxList = itemList[a]["bbox"]

            cv2.rectangle(imageDebug, (bboxList[0], bboxList[1]), (bboxList[2], bboxList[3]), (0, 200, 0), 1)

        cv2.imwrite(f"{pathOutput}debug/ocr/{numberPage}.jpg", imageDebug)

    def execute(self, pathInput, pathOutput):
        pageList = self._pageBuild(pathInput, pathOutput)

        pageReaderList = self.reader.execute(pathInput)

        astPageList = []
        layoutList = []
        tableList = []
        itemList = []

        for a in range(len(pageList)):
            astPage = self.layout.execute(pathOutput, pageList[a]["image"], pageList[a]["number"])

            astPageList.append(astPage)

            tablePageList = self.table.execute(astPage, pageList[a]["image"])

            itemPageList = self._itemBuild(pageList[a]["image"], pageReaderList[a], len(itemList), pageList[a]["number"])

            self._debugText(pageList[a]["image"], itemPageList, pathOutput, pageList[a]["number"])

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

    def __init__(self, layout, table):
        self.reader = Reader()
        self.layout = layout
        self.table = table

class Reader:
    def _byteText(self, byteList):
        return byteList.decode("latin-1")

    def _parseIndirect(self):
        resultList = []

        matchList = list(re.finditer(r"(\d+)\s+(\d+)\s+obj\b", self.text))

        for a in range(len(matchList)):
            self.position = matchList[a].end()

            value = self._parseValue()

            category = value["kind"]

            if (value["kind"] == "dictionary" or value["kind"] == "stream") and value.get("category") is not None:
                category = value["category"]

            resultList.append({
                "number": int(matchList[a].group(1)),
                "generation": int(matchList[a].group(2)),
                "category": category,
                "value": value
            })

        expandedList = []

        for a in range(len(resultList)):
            nestedList = self._streamIndirectExpand(resultList[a])

            for b in range(len(nestedList)):
                expandedList.append(nestedList[b])

        for a in range(len(expandedList)):
            resultList.append(expandedList[a])

        return resultList

    def _parseValue(self):
        self._skipWhitespace()

        code = self.byteList[self.position] if self.position < len(self.byteList) else 0

        if code == 47:
            result = self._parseName()
        elif code == 40:
            result = self._parseLiteralString()
        elif code == 60 and self.position + 1 < len(self.byteList) and self.byteList[self.position + 1] == 60:
            result = self._parseDictionaryOrStream()
        elif code == 60:
            result = self._parseHexString()
        elif code == 91:
            result = self._parseArray()
        elif self._digitCheck(code) or code == 43 or code == 45 or code == 46:
            result = self._parseNumberOrReference()
        elif self.text[self.position:self.position + 4] == "true":
            self.position += 4
            result = {"kind": "boolean", "value": True}
        elif self.text[self.position:self.position + 5] == "false":
            self.position += 5
            result = {"kind": "boolean", "value": False}
        elif self.text[self.position:self.position + 4] == "null":
            self.position += 4
            result = {"kind": "null"}
        else:
            operator = ""

            while (
                self.position < len(self.byteList)
                and self._whitespaceCheck(self.byteList[self.position]) == False
                and self._delimiterCheck(self.byteList[self.position]) == False
            ):
                operator += chr(self.byteList[self.position])
                self.position += 1

            result = {"kind": "operator", "value": operator}

        return result

    def _skipWhitespace(self):
        byteList = self.byteList
        length = len(byteList)

        isRunning = True

        while isRunning:
            if self.position >= length:
                isRunning = False
            else:
                code = byteList[self.position]

                if code in self.whitespaceSet:
                    self.position += 1
                elif code == 37:
                    while self.position < length and byteList[self.position] != 10 and byteList[self.position] != 13:
                        self.position += 1
                else:
                    isRunning = False

    def _parseName(self):
        self.position += 1

        value = ""
        isRunning = True

        while isRunning:
            if self.position >= len(self.byteList):
                isRunning = False
            else:
                code = self.byteList[self.position]

                if self._whitespaceCheck(code) or self._delimiterCheck(code):
                    isRunning = False
                elif code == 35:
                    hexText = self.text[self.position + 1:self.position + 3]

                    if re.fullmatch(r"[0-9A-Fa-f]{2}", hexText) is not None:
                        value += chr(int(hexText, 16))

                    self.position += 3
                else:
                    value += chr(code)
                    self.position += 1

        return {"kind": "name", "value": value}

    def _whitespaceCheck(self, code):
        return code in self.whitespaceSet

    def _delimiterCheck(self, code):
        return code in self.delimiterSet

    def _parseLiteralString(self):
        self.position += 1

        value = ""
        depth = 1

        while depth > 0 and self.position < len(self.byteList):
            code = self.byteList[self.position]

            if code == 92:
                nextCode = self.byteList[self.position + 1] if self.position + 1 < len(self.byteList) else 0

                if nextCode == 110:
                    value += "\n"
                    self.position += 2
                elif nextCode == 114:
                    value += "\r"
                    self.position += 2
                elif nextCode == 116:
                    value += "\t"
                    self.position += 2
                elif nextCode >= 48 and nextCode <= 55:
                    octalText = ""
                    count = 0

                    self.position += 1

                    while count < 3 and self.position < len(self.byteList) and self.byteList[self.position] >= 48 and self.byteList[self.position] <= 55:
                        octalText += chr(self.byteList[self.position])
                        self.position += 1
                        count += 1

                    if len(octalText) > 0:
                        value += chr(int(octalText, 8) & 0xff)
                else:
                    value += chr(nextCode)
                    self.position += 2
            elif code == 40:
                depth += 1
                value += "("
                self.position += 1
            elif code == 41:
                depth -= 1

                if depth > 0:
                    value += ")"

                self.position += 1
            else:
                value += chr(code)
                self.position += 1

        return {"kind": "string", "value": value}

    def _parseDictionaryOrStream(self):
        self.position += 2

        entryObject = {}

        isRunning = True

        while isRunning:
            self._skipWhitespace()

            if self.position >= len(self.byteList):
                isRunning = False
            elif self.byteList[self.position] == 62 and self.position + 1 < len(self.byteList) and self.byteList[self.position + 1] == 62:
                self.position += 2
                isRunning = False
            elif self.byteList[self.position] == 47:
                nameNode = self._parseName()

                self._skipWhitespace()

                entryObject[nameNode["value"]] = self._parseValue()
            else:
                isRunning = False

        category = self._dictionaryCategory(entryObject)

        self._skipWhitespace()

        result = {"kind": "dictionary", "category": category, "entryObject": entryObject}

        if self.text[self.position:self.position + 6] == "stream":
            result = self._parseStream(entryObject, category)

        return result

    def _dictionaryCategory(self, entryObject):
        result = "dictionary"

        typeNode = entryObject.get("Type")

        if typeNode is not None and typeNode["kind"] == "name":
            result = typeNode["value"]

            subtypeNode = entryObject.get("Subtype")

            if subtypeNode is not None and subtypeNode["kind"] == "name":
                result = f"{typeNode['value']}:{subtypeNode['value']}"

        return result

    def _parseStream(self, entryObject, category):
        self.position += 6

        if self.position < len(self.byteList) and self.byteList[self.position] == 13:
            self.position += 1

        if self.position < len(self.byteList) and self.byteList[self.position] == 10:
            self.position += 1

        start = self.position
        endIndex = self.text.find("endstream", start)

        if endIndex < 0:
            endIndex = len(self.byteList)

        end = endIndex

        if end - 1 >= 0 and end - 1 < len(self.byteList) and self.byteList[end - 1] == 10:
            end -= 1

        if end - 1 >= 0 and end - 1 < len(self.byteList) and self.byteList[end - 1] == 13:
            end -= 1

        rawList = self.byteList[start:end]

        self.position = endIndex + 9

        filterList = self._filterExtract(entryObject)
        isImage = "Image" in category or "DCTDecode" in filterList or "JPXDecode" in filterList

        result = {
            "kind": "stream",
            "category": category,
            "entryObject": entryObject,
            "filterList": filterList,
            "rawByteLength": len(rawList)
        }

        if isImage:
            result["isImage"] = True
        else:
            decodedList = self._decodeStream(rawList, filterList, entryObject)

            result["decodedByteLength"] = len(decodedList)
            result["content"] = self._byteText(decodedList)

        return result

    def _filterExtract(self, entryObject):
        resultList = []

        filterNode = entryObject.get("Filter")

        if filterNode is not None:
            if filterNode["kind"] == "name":
                resultList.append(filterNode["value"])
            elif filterNode["kind"] == "array" and filterNode.get("itemList") is not None:
                for a in range(len(filterNode["itemList"])):
                    item = filterNode["itemList"][a]

                    if item["kind"] == "name":
                        resultList.append(item["value"])

        return resultList

    def _decodeStream(self, rawList, filterList, entryObject):
        result = rawList

        for a in range(len(filterList)):
            if filterList[a] == "FlateDecode" or filterList[a] == "Fl":
                result = self._inflate(result)
                result = self._applyPredictor(result, entryObject)

        return result

    def _inflate(self, byteList):
        isZlibHeader = False

        if len(byteList) >= 2:
            byte0 = byteList[0]
            byte1 = byteList[1]

            isZlibHeader = (byte0 & 0x0f) == 8 and ((byte0 << 8) | byte1) % 31 == 0

        decompressor = zlib.decompressobj() if isZlibHeader else zlib.decompressobj(-15)

        return decompressor.decompress(bytes(byteList))

    def _applyPredictor(self, byteList, entryObject):
        result = byteList

        parmsNode = entryObject.get("DecodeParms")

        if parmsNode is not None and parmsNode["kind"] == "dictionary" and parmsNode.get("entryObject") is not None:
            predictorNode = parmsNode["entryObject"].get("Predictor")
            columnsNode = parmsNode["entryObject"].get("Columns")

            if predictorNode is not None and predictorNode["kind"] == "number" and predictorNode["value"] >= 10:
                columns = int(columnsNode["value"]) if columnsNode is not None and columnsNode["kind"] == "number" else 1

                result = self._applyPngPredictor(columns, byteList)

        return result

    def _applyPngPredictor(self, columns, byteList):
        rowLength = columns + 1
        rowCount = len(byteList) // rowLength

        resultList = bytearray(rowCount * columns)

        previousList = bytearray(columns)

        for row in range(rowCount):
            filterType = byteList[row * rowLength]
            currentList = bytearray(columns)

            for a in range(columns):
                value = byteList[row * rowLength + 1 + a]
                left = currentList[a - 1] if a >= 1 else 0
                up = previousList[a]
                upLeft = previousList[a - 1] if a >= 1 else 0

                restored = value

                if filterType == 1:
                    restored = value + left
                elif filterType == 2:
                    restored = value + up
                elif filterType == 3:
                    restored = value + (left + up) // 2
                elif filterType == 4:
                    paeth = left + up - upLeft
                    paethLeft = abs(paeth - left)
                    paethUp = abs(paeth - up)
                    paethUpLeft = abs(paeth - upLeft)

                    predictor = upLeft

                    if paethLeft <= paethUp and paethLeft <= paethUpLeft:
                        predictor = left
                    elif paethUp <= paethUpLeft:
                        predictor = up

                    restored = value + predictor

                currentList[a] = restored & 0xff
                resultList[row * columns + a] = currentList[a]

            previousList = currentList

        return bytes(resultList)

    def _parseHexString(self):
        self.position += 1

        byteList = self.byteList
        length = len(byteList)

        startPosition = self.position

        while self.position < length and byteList[self.position] != 62:
            self.position += 1

        hexText = re.sub(r"[^0-9A-Fa-f]", "", self.text[startPosition:self.position])

        self.position += 1

        if len(hexText) % 2 == 1:
            hexText += "0"

        value = bytes.fromhex(hexText).decode("latin-1")

        return {"kind": "hexString", "value": value}

    def _parseArray(self):
        self.position += 1

        itemList = []

        isRunning = True

        while isRunning:
            self._skipWhitespace()

            if self.position >= len(self.byteList) or self.byteList[self.position] == 93:
                self.position += 1
                isRunning = False
            else:
                itemList.append(self._parseValue())

        return {"kind": "array", "itemList": itemList}

    def _digitCheck(self, code):
        return code >= 48 and code <= 57

    def _parseNumberOrReference(self):
        savedPosition = self.position

        byteList = self.byteList
        length = len(byteList)

        isRunning = True

        while isRunning:
            if self.position >= length:
                isRunning = False
            else:
                code = byteList[self.position]

                if (code >= 48 and code <= 57) or code == 43 or code == 45 or code == 46:
                    self.position += 1
                else:
                    isRunning = False

        numberText = self.text[savedPosition:self.position]

        firstNumber = self._floatParse(numberText)

        result = {"kind": "number", "value": firstNumber}

        if "." not in numberText:
            afterFirst = self.position

            self._skipWhitespace()

            secondPosition = self.position

            while self.position < length and byteList[self.position] >= 48 and byteList[self.position] <= 57:
                self.position += 1

            secondText = self.text[secondPosition:self.position]

            if len(secondText) > 0:
                self._skipWhitespace()

                if self.position < len(self.byteList) and self.byteList[self.position] == 82:
                    self.position += 1
                    result = {"kind": "reference", "number": int(firstNumber), "generation": int(secondText)}
                else:
                    self.position = afterFirst
            else:
                self.position = afterFirst

        if result["kind"] == "number":
            self.position = savedPosition + len(numberText)

        return result

    def _floatParse(self, text):
        result = 0.0

        match = re.match(r"[+-]?(\d+\.?\d*|\.\d+)", text)

        if match is not None:
            result = float(match.group(0))

        return result

    def _streamIndirectExpand(self, indirect):
        resultList = []

        streamNode = indirect["value"]

        if streamNode["kind"] == "stream" and streamNode.get("category") == "ObjStm" and streamNode.get("content") is not None and streamNode.get("entryObject") is not None:
            countNode = streamNode["entryObject"].get("N")
            firstNode = streamNode["entryObject"].get("First")

            if countNode is not None and countNode["kind"] == "number" and firstNode is not None and firstNode["kind"] == "number":
                count = int(countNode["value"])
                first = int(firstNode["value"])

                savedByteList = self.byteList
                savedText = self.text
                savedPosition = self.position

                self.text = streamNode["content"]
                self.byteList = self._textByte(streamNode["content"])

                headerList = []

                self.position = 0

                for a in range(count):
                    self._skipWhitespace()
                    numberNode = self._parseValue()
                    self._skipWhitespace()
                    offsetNode = self._parseValue()

                    if numberNode["kind"] == "number" and offsetNode["kind"] == "number":
                        headerList.append({"number": int(numberNode["value"]), "offset": int(offsetNode["value"])})

                for a in range(len(headerList)):
                    self.position = first + headerList[a]["offset"]

                    value = self._parseValue()

                    category = value["kind"]

                    if (value["kind"] == "dictionary" or value["kind"] == "stream") and value.get("category") is not None:
                        category = value["category"]

                    resultList.append({"number": headerList[a]["number"], "generation": 0, "category": category, "value": value})

                self.byteList = savedByteList
                self.text = savedText
                self.position = savedPosition

        return resultList

    def _textByte(self, text):
        return text.encode("latin-1", errors="replace")

    def _buildPage(self):
        resultList = []

        trailerIndex = self.text.rfind("trailer")

        rootNode = None

        if trailerIndex >= 0:
            self.position = trailerIndex + 7
            self._skipWhitespace()

            trailer = self._parseValue()

            if trailer.get("entryObject") is not None:
                rootNode = trailer["entryObject"].get("Root")

        if rootNode is None:
            indirectList = list(self.indirectObject.values())

            for a in range(len(indirectList)):
                if indirectList[a]["category"] == "Catalog":
                    rootNode = indirectList[a]["value"]

        catalog = self._resolve(rootNode)
        pageRawList = []

        if catalog is not None and catalog.get("entryObject") is not None:
            self._collectPage(catalog["entryObject"].get("Pages"), {}, [0, 0, 595, 842], pageRawList)

        for a in range(len(pageRawList)):
            pageRaw = pageRawList[a]

            self.ctmList = [1, 0, 0, 1, 0, 0]
            self.textMatrixList = [1, 0, 0, 1, 0, 0]
            self.lineMatrixList = [1, 0, 0, 1, 0, 0]
            self.graphicsStateList = []
            self.fontSize = 0
            self.charSpacing = 0
            self.wordSpacing = 0
            self.horizontalScale = 1
            self.leading = 0
            self.textRender = 0
            self.textRise = 0
            self.fillColor = "#000000"
            self.strokeColor = "#000000"
            self.currentFont = None
            self._pathReset()

            width = pageRaw["mediaBoxList"][2] - pageRaw["mediaBoxList"][0]
            height = pageRaw["mediaBoxList"][3] - pageRaw["mediaBoxList"][1]

            self.pageHeight = height
            self.elementList = []

            content = self._pageContent(pageRaw["entryObject"])

            self._interpretContent(content, pageRaw["resourceObject"])
            self._pageLink(pageRaw["entryObject"])

            resultList.append({"number": a + 1, "width": width, "height": height, "elementList": self._mergeText(self.elementList)})

        return resultList

    def _resolve(self, node):
        result = node

        while result is not None and result["kind"] == "reference":
            found = self.indirectObject.get(result["number"])

            result = found["value"] if found is not None else None

        return result

    def _collectPage(self, node, parentResourceObject, parentMediaBoxList, resultList):
        resolved = self._resolve(node)

        if resolved is not None and resolved.get("entryObject") is not None:
            resourceObject = parentResourceObject
            mediaBoxList = parentMediaBoxList

            resourceNode = self._resolve(resolved["entryObject"].get("Resources"))

            if resourceNode is not None and resourceNode.get("entryObject") is not None:
                resourceObject = resourceNode["entryObject"]

            mediaBoxNode = self._resolve(resolved["entryObject"].get("MediaBox"))

            if mediaBoxNode is not None and mediaBoxNode["kind"] == "array" and mediaBoxNode.get("itemList") is not None:
                itemList = mediaBoxNode["itemList"]

                mediaBoxList = [
                    self._numberValue(itemList[0] if len(itemList) > 0 else None),
                    self._numberValue(itemList[1] if len(itemList) > 1 else None),
                    self._numberValue(itemList[2] if len(itemList) > 2 else None),
                    self._numberValue(itemList[3] if len(itemList) > 3 else None)
                ]

            typeNode = self._resolve(resolved["entryObject"].get("Type"))
            type = typeNode["value"] if typeNode is not None and typeNode["kind"] == "name" else ""

            if type == "Page":
                resultList.append({"entryObject": resolved["entryObject"], "resourceObject": resourceObject, "mediaBoxList": mediaBoxList})
            else:
                kidsNode = self._resolve(resolved["entryObject"].get("Kids"))

                if kidsNode is not None and kidsNode["kind"] == "array" and kidsNode.get("itemList") is not None:
                    for a in range(len(kidsNode["itemList"])):
                        self._collectPage(kidsNode["itemList"][a], resourceObject, mediaBoxList, resultList)

    def _numberValue(self, node):
        result = 0

        resolved = self._resolve(node)

        if resolved is not None and resolved["kind"] == "number":
            result = resolved["value"]

        return result

    def _pathReset(self):
        self.isPathEmpty = True
        self.isPathRectangle = False

    def _pageContent(self, entryObject):
        result = ""

        contentNode = self._resolve(entryObject.get("Contents"))

        if contentNode is not None:
            if contentNode["kind"] == "stream" and contentNode.get("content") is not None:
                result = contentNode["content"]
            elif contentNode["kind"] == "array" and contentNode.get("itemList") is not None:
                for a in range(len(contentNode["itemList"])):
                    part = self._resolve(contentNode["itemList"][a])

                    if part is not None and part["kind"] == "stream" and part.get("content") is not None:
                        result += f"{part['content']}\n"

        return result

    def _interpretContent(self, content, resourceObject):
        fontObject = self._resourceFont(resourceObject)
        externalObject = self._resourceExternal(resourceObject)
        stateObject = self._resourceState(resourceObject)

        self.byteList = self._textByte(content)
        self.text = content
        self.position = 0

        stackList = []

        while self.position < len(self.byteList):
            self._skipWhitespace()

            if self.position >= len(self.byteList):
                break

            node = self._parseValue()

            if node["kind"] == "operator":
                if len(node["value"]) > 0:
                    self._handleOperator(node["value"], stackList, stateObject, fontObject, externalObject)
                else:
                    self.position += 1

                stackList = []
            else:
                stackList.append(node)

    def _resourceFont(self, resourceObject):
        resultObject = {}

        fontNode = self._resolve(resourceObject.get("Font"))

        if fontNode is not None and fontNode["kind"] == "dictionary" and fontNode.get("entryObject") is not None:
            nameList = list(fontNode["entryObject"].keys())

            for a in range(len(nameList)):
                resolved = self._resolve(fontNode["entryObject"][nameList[a]])

                if resolved is not None:
                    resultObject[nameList[a]] = self._buildFont(resolved)

        return resultObject

    def _buildFont(self, fontNode):
        entryObject = fontNode["entryObject"] if fontNode.get("entryObject") is not None else {}

        baseFontNode = self._resolve(entryObject.get("BaseFont"))
        baseFont = baseFontNode["value"] if baseFontNode is not None and baseFontNode["kind"] == "name" else ""

        subtypeNode = self._resolve(entryObject.get("Subtype"))
        subtype = subtypeNode["value"] if subtypeNode is not None and subtypeNode["kind"] == "name" else ""

        encodingNode = self._resolve(entryObject.get("Encoding"))
        encoding = encodingNode["value"] if encodingNode is not None and encodingNode["kind"] == "name" else ""

        codecName = self._codecGet(encoding) if subtype == "Type0" else ""

        isBold = "bold" in baseFont.lower()
        byteLength = 2 if subtype == "Type0" else 1

        result = {
            "baseFont": baseFont,
            "isBold": isBold,
            "byteLength": byteLength,
            "firstChar": 0,
            "widthList": [],
            "widthScale": 0.001,
            "widthObject": {},
            "defaultWidthFraction": 0.5,
            "codecName": codecName,
            "isUnicodeCode": codecName == "utf-16-be",
            "encodingObject": self._buildEncoding(encodingNode) if subtype != "Type0" else {},
            "toUnicodeObject": {}
        }

        toUnicodeNode = self._resolve(entryObject.get("ToUnicode"))

        if toUnicodeNode is not None and toUnicodeNode["kind"] == "stream" and toUnicodeNode.get("content") is not None:
            result["toUnicodeObject"] = self._buildToUnicode(toUnicodeNode["content"])

        if subtype == "Type0":
            descendantNode = self._resolve(entryObject.get("DescendantFonts"))

            if descendantNode is not None and descendantNode["kind"] == "array" and descendantNode.get("itemList") is not None:
                cidFontNode = self._resolve(descendantNode["itemList"][0] if len(descendantNode["itemList"]) > 0 else None)

                if cidFontNode is not None and cidFontNode.get("entryObject") is not None:
                    defaultWidthNode = self._resolve(cidFontNode["entryObject"].get("DW"))

                    result["defaultWidthFraction"] = defaultWidthNode["value"] / 1000 if defaultWidthNode is not None and defaultWidthNode["kind"] == "number" else 1
                    result["widthObject"] = self._cidWidth(cidFontNode["entryObject"])
        else:
            firstCharNode = self._resolve(entryObject.get("FirstChar"))

            if firstCharNode is not None and firstCharNode["kind"] == "number":
                result["firstChar"] = int(firstCharNode["value"])

            widthsNode = self._resolve(entryObject.get("Widths"))

            if widthsNode is not None and widthsNode["kind"] == "array" and widthsNode.get("itemList") is not None:
                for a in range(len(widthsNode["itemList"])):
                    result["widthList"].append(self._numberValue(widthsNode["itemList"][a]))

            if subtype == "Type3":
                matrixNode = self._resolve(entryObject.get("FontMatrix"))

                if matrixNode is not None and matrixNode["kind"] == "array" and matrixNode.get("itemList") is not None:
                    result["widthScale"] = self._numberValue(matrixNode["itemList"][0] if len(matrixNode["itemList"]) > 0 else None)

        return result

    def _codecGet(self, encoding):
        for a in range(len(self.codecList)):
            if self.codecList[a][0] in encoding:
                return self.codecList[a][1]

        return ""

    def _buildEncoding(self, encodingNode):
        resultObject = {}

        baseName = "/StandardEncoding"
        differenceNode = None

        if encodingNode is not None and encodingNode["kind"] == "name":
            baseName = encodingNode["value"]
        elif encodingNode is not None and encodingNode["kind"] == "dictionary" and encodingNode.get("entryObject") is not None:
            baseNode = self._resolve(encodingNode["entryObject"].get("BaseEncoding"))

            if baseNode is not None and baseNode["kind"] == "name":
                baseName = baseNode["value"]

            differenceNode = self._resolve(encodingNode["entryObject"].get("Differences"))

        codecName = "cp1252" if "WinAnsi" in baseName else "mac_roman" if "MacRoman" in baseName else "latin-1"

        for a in range(32, 256):
            character = bytes([a]).decode(codecName, errors="ignore")

            if len(character) > 0:
                resultObject[a] = character

        if differenceNode is not None and differenceNode["kind"] == "array" and differenceNode.get("itemList") is not None:
            code = 0

            for a in range(len(differenceNode["itemList"])):
                item = self._resolve(differenceNode["itemList"][a])

                if item is None:
                    continue

                if item["kind"] == "number":
                    code = int(item["value"])
                elif item["kind"] == "name":
                    character = self._glyphUnicode(item["value"])

                    if len(character) > 0:
                        resultObject[code] = character

                    code += 1

        return resultObject

    def _glyphUnicode(self, name):
        if name[0:3] == "uni" and len(name) >= 7:
            return chr(int(name[3:7], 16))

        if name[0:1] == "u" and len(name) >= 5 and len(name) <= 7:
            return chr(int(name[1:], 16))

        if len(name) == 1:
            return name

        if name in self.glyphObject:
            return self.glyphObject[name]

        return ""

    def _buildToUnicode(self, content):
        resultObject = {}

        charBlockList = list(re.finditer(r"beginbfchar([\s\S]*?)endbfchar", content))

        for a in range(len(charBlockList)):
            pairList = list(re.finditer(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", charBlockList[a].group(1)))

            for b in range(len(pairList)):
                resultObject[int(pairList[b].group(1), 16)] = self._utf16Hex(pairList[b].group(2))

        rangeBlockList = list(re.finditer(r"beginbfrange([\s\S]*?)endbfrange", content))

        for a in range(len(rangeBlockList)):
            lineList = list(re.finditer(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*(\[[\s\S]*?\]|<[0-9A-Fa-f]+>)", rangeBlockList[a].group(1)))

            for b in range(len(lineList)):
                low = int(lineList[b].group(1), 16)
                high = int(lineList[b].group(2), 16)
                destination = lineList[b].group(3)

                if destination[0:1] == "[":
                    itemList = list(re.finditer(r"<([0-9A-Fa-f]+)>", destination))
                    code = low

                    for c in range(len(itemList)):
                        if code <= high:
                            resultObject[code] = self._utf16Hex(itemList[c].group(1))
                            code += 1
                else:
                    base = int(destination.replace("<", "").replace(">", ""), 16)

                    for c in range(high - low + 1):
                        resultObject[low + c] = chr((base + c) & 0xffff)

        return resultObject

    def _utf16Hex(self, hexText):
        result = ""

        for a in range(0, len(hexText) - 3, 4):
            result += chr(int(hexText[a:a + 4], 16))

        return result

    def _cidWidth(self, cidFontObject):
        resultObject = {}

        widthNode = self._resolve(cidFontObject.get("W"))

        if widthNode is not None and widthNode["kind"] == "array" and widthNode.get("itemList") is not None:
            itemList = widthNode["itemList"]

            a = 0

            while a < len(itemList):
                first = self._numberValue(itemList[a])
                second = self._resolve(itemList[a + 1]) if a + 1 < len(itemList) else None

                if second is not None and second["kind"] == "array" and second.get("itemList") is not None:
                    for b in range(len(second["itemList"])):
                        resultObject[int(first) + b] = self._numberValue(second["itemList"][b]) / 1000

                    a += 2
                else:
                    last = self._numberValue(itemList[a + 1]) if a + 1 < len(itemList) else 0
                    width = self._numberValue(itemList[a + 2]) / 1000 if a + 2 < len(itemList) else 0

                    for cid in range(int(first), int(last) + 1):
                        resultObject[cid] = width

                    a += 3

        return resultObject

    def _resourceExternal(self, resourceObject):
        resultObject = {}

        externalNode = self._resolve(resourceObject.get("XObject"))

        if externalNode is not None and externalNode["kind"] == "dictionary" and externalNode.get("entryObject") is not None:
            nameList = list(externalNode["entryObject"].keys())

            for a in range(len(nameList)):
                reference = externalNode["entryObject"][nameList[a]]
                resolved = self._resolve(reference)

                if resolved is not None and resolved.get("entryObject") is not None:
                    subtypeNode = self._resolve(resolved["entryObject"].get("Subtype"))

                    resultObject[nameList[a]] = {
                        "referenceNumber": reference["number"] if reference["kind"] == "reference" else 0,
                        "subtype": subtypeNode["value"] if subtypeNode is not None and subtypeNode["kind"] == "name" else "",
                        "width": self._numberValue(resolved["entryObject"].get("Width")),
                        "height": self._numberValue(resolved["entryObject"].get("Height"))
                    }

        return resultObject

    def _resourceState(self, resourceObject):
        resultObject = {}

        stateNode = self._resolve(resourceObject.get("ExtGState"))

        if stateNode is not None and stateNode["kind"] == "dictionary" and stateNode.get("entryObject") is not None:
            nameList = list(stateNode["entryObject"].keys())

            for a in range(len(nameList)):
                resolved = self._resolve(stateNode["entryObject"][nameList[a]])

                if resolved is not None and resolved.get("entryObject") is not None:
                    fontNode = self._resolve(resolved["entryObject"].get("Font"))

                    if fontNode is not None and fontNode["kind"] == "array" and fontNode.get("itemList") is not None and len(fontNode["itemList"]) == 2:
                        fontResolved = self._resolve(fontNode["itemList"][0])

                        if fontResolved is not None:
                            resultObject[nameList[a]] = {
                                "font": self._buildFont(fontResolved),
                                "fontSize": self._numberValue(fontNode["itemList"][1])
                            }

        return resultObject

    def _handleOperator(self, operator, stackList, stateObject, fontObject, externalObject):
        def number(indexFromEnd):
            result = 0

            if indexFromEnd >= 1 and indexFromEnd <= len(stackList):
                node = stackList[len(stackList) - indexFromEnd]

                if node["kind"] == "number":
                    result = node["value"]

            return result

        if operator == "cm":
            self.ctmList = self._matrixMultiply(self.ctmList, [number(6), number(5), number(4), number(3), number(2), number(1)])
        elif operator == "q":
            self.graphicsStateList.append({
                "ctmList": list(self.ctmList),
                "currentFont": self.currentFont,
                "fontSize": self.fontSize,
                "charSpacing": self.charSpacing,
                "wordSpacing": self.wordSpacing,
                "horizontalScale": self.horizontalScale,
                "leading": self.leading,
                "textRender": self.textRender,
                "textRise": self.textRise,
                "fillColor": self.fillColor,
                "strokeColor": self.strokeColor
            })
        elif operator == "Q":
            if len(self.graphicsStateList) > 0:
                stateObject = self.graphicsStateList.pop()

                self.ctmList = stateObject["ctmList"]
                self.currentFont = stateObject["currentFont"]
                self.fontSize = stateObject["fontSize"]
                self.charSpacing = stateObject["charSpacing"]
                self.wordSpacing = stateObject["wordSpacing"]
                self.horizontalScale = stateObject["horizontalScale"]
                self.leading = stateObject["leading"]
                self.textRender = stateObject["textRender"]
                self.textRise = stateObject["textRise"]
                self.fillColor = stateObject["fillColor"]
                self.strokeColor = stateObject["strokeColor"]
        elif operator == "BT":
            self.textMatrixList = [1, 0, 0, 1, 0, 0]
            self.lineMatrixList = [1, 0, 0, 1, 0, 0]
        elif operator == "Tf":
            nameNode = stackList[len(stackList) - 2] if len(stackList) >= 2 else None

            self.fontSize = number(1)

            if nameNode is not None and nameNode["kind"] == "name":
                self.currentFont = fontObject.get(nameNode["value"])
        elif operator == "Tr":
            self.textRender = int(number(1))
        elif operator == "Ts":
            self.textRise = number(1)
        elif operator == "gs":
            nameNode = stackList[len(stackList) - 1] if len(stackList) >= 1 else None

            if nameNode is not None and nameNode["kind"] == "name" and nameNode["value"] in stateObject:
                self.currentFont = stateObject[nameNode["value"]]["font"]
                self.fontSize = stateObject[nameNode["value"]]["fontSize"]
        elif operator == "Td":
            self.lineMatrixList = self._matrixMultiply(self.lineMatrixList, [1, 0, 0, 1, number(2), number(1)])
            self.textMatrixList = list(self.lineMatrixList)
        elif operator == "TD":
            self.leading = -number(1)
            self.lineMatrixList = self._matrixMultiply(self.lineMatrixList, [1, 0, 0, 1, number(2), number(1)])
            self.textMatrixList = list(self.lineMatrixList)
        elif operator == "Tm":
            self.lineMatrixList = [number(6), number(5), number(4), number(3), number(2), number(1)]
            self.textMatrixList = list(self.lineMatrixList)
        elif operator == "T*":
            self.lineMatrixList = self._matrixMultiply(self.lineMatrixList, [1, 0, 0, 1, 0, -self.leading])
            self.textMatrixList = list(self.lineMatrixList)
        elif operator == "Tc":
            self.charSpacing = number(1)
        elif operator == "Tw":
            self.wordSpacing = number(1)
        elif operator == "Tz":
            self.horizontalScale = number(1) / 100
        elif operator == "TL":
            self.leading = number(1)
        elif operator == "Tj" and self.currentFont is not None:
            if len(stackList) > 0:
                self._showText([stackList[len(stackList) - 1]], self.currentFont)
        elif operator == "TJ" and self.currentFont is not None:
            arrayNode = stackList[len(stackList) - 1] if len(stackList) > 0 else None

            if arrayNode is not None and arrayNode["kind"] == "array" and arrayNode.get("itemList") is not None:
                self._showText(arrayNode["itemList"], self.currentFont)
        elif (operator == "'" or operator == '"') and self.currentFont is not None:
            self.lineMatrixList = self._matrixMultiply(self.lineMatrixList, [1, 0, 0, 1, 0, -self.leading])
            self.textMatrixList = list(self.lineMatrixList)

            if len(stackList) > 0:
                self._showText([stackList[len(stackList) - 1]], self.currentFont)
        elif operator == "g":
            self.fillColor = self._colorRgb(number(1), number(1), number(1))
        elif operator == "G":
            self.strokeColor = self._colorRgb(number(1), number(1), number(1))
        elif operator == "rg":
            self.fillColor = self._colorRgb(number(3), number(2), number(1))
        elif operator == "RG":
            self.strokeColor = self._colorRgb(number(3), number(2), number(1))
        elif operator == "k":
            self.fillColor = self._colorRgb((1 - number(4)) * (1 - number(1)), (1 - number(3)) * (1 - number(1)), (1 - number(2)) * (1 - number(1)))
        elif operator == "K":
            self.strokeColor = self._colorRgb((1 - number(4)) * (1 - number(1)), (1 - number(3)) * (1 - number(1)), (1 - number(2)) * (1 - number(1)))
        elif operator == "m" or operator == "l":
            self._pathAddPoint(number(2), number(1))
        elif operator == "c":
            self._pathAddPoint(number(2), number(1))
        elif operator == "v" or operator == "y":
            self._pathAddPoint(number(2), number(1))
        elif operator == "re":
            x = number(4)
            y = number(3)
            width = number(2)
            height = number(1)
            wasEmpty = self.isPathEmpty

            self._pathAddPoint(x, y)
            self._pathAddPoint(x + width, y + height)

            self.isPathRectangle = wasEmpty
        elif operator == "f" or operator == "F" or operator == "f*":
            self._pathPaint(True, False)
        elif operator == "S" or operator == "s":
            self._pathPaint(False, True)
        elif operator == "B" or operator == "B*" or operator == "b" or operator == "b*":
            self._pathPaint(True, True)
        elif operator == "n":
            self._pathReset()
        elif operator == "Do":
            nameNode = stackList[len(stackList) - 1] if len(stackList) > 0 else None

            if nameNode is not None and nameNode["kind"] == "name":
                external = externalObject.get(nameNode["value"])

                if external is not None and external["subtype"] == "Image":
                    cornerAList = self._transformPoint(self.ctmList, 0, 0)
                    cornerBList = self._transformPoint(self.ctmList, 1, 1)

                    self.elementList.append({
                        "type": "image",
                        "x0": min(cornerAList[0], cornerBList[0]),
                        "y0": self.pageHeight - max(cornerAList[1], cornerBList[1]),
                        "x1": max(cornerAList[0], cornerBList[0]),
                        "y1": self.pageHeight - min(cornerAList[1], cornerBList[1]),
                        "referenceNumber": external["referenceNumber"]
                    })
        elif operator == "BI":
            endIndex = self.text.find("EI", self.position)

            self.position = endIndex + 2 if endIndex >= 0 else len(self.byteList)

    def _matrixMultiply(self, rightList, leftList):
        return [
            leftList[0] * rightList[0] + leftList[1] * rightList[2],
            leftList[0] * rightList[1] + leftList[1] * rightList[3],
            leftList[2] * rightList[0] + leftList[3] * rightList[2],
            leftList[2] * rightList[1] + leftList[3] * rightList[3],
            leftList[4] * rightList[0] + leftList[5] * rightList[2] + rightList[4],
            leftList[4] * rightList[1] + leftList[5] * rightList[3] + rightList[5]
        ]

    def _showText(self, partList, font):
        text = ""
        advance = 0

        for a in range(len(partList)):
            part = partList[a]

            if part["kind"] == "string" or part["kind"] == "hexString":
                decoded = self._fontDecode(part["value"], font)

                for b in range(len(decoded["charList"])):
                    text += decoded["charList"][b]

                    glyph = decoded["widthFractionList"][b] * self.fontSize + self.charSpacing

                    if font["byteLength"] == 1 and decoded["codeList"][b] == 32:
                        glyph += self.wordSpacing

                    advance += glyph * self.horizontalScale
            elif part["kind"] == "number":
                advance -= part["value"] / 1000 * self.fontSize * self.horizontalScale

        renderMatrixList = self._matrixMultiply(self.ctmList, self.textMatrixList)
        deviceFontSize = self.fontSize * math.hypot(renderMatrixList[2], renderMatrixList[3])

        startList = self._transformPoint(renderMatrixList, 0, self.textRise)
        endList = self._transformPoint(renderMatrixList, advance, self.textRise)

        if len(text.strip()) > 0 and self.textRender != 3 and self.textRender != 7:
            self.elementList.append({
                "type": "text",
                "text": text,
                "x0": min(startList[0], endList[0]),
                "y0": self.pageHeight - (startList[1] + deviceFontSize * 0.8),
                "x1": max(startList[0], endList[0]),
                "y1": self.pageHeight - (startList[1] - deviceFontSize * 0.2),
                "fontName": font["baseFont"],
                "fontSize": math.floor(deviceFontSize * 100 + 0.5) / 100,
                "isBold": font["isBold"],
                "color": self.fillColor
            })

        self.textMatrixList = self._matrixMultiply(self.textMatrixList, [1, 0, 0, 1, advance, 0])

    def _fontDecode(self, raw, font):
        charList = []
        widthFractionList = []
        codeList = []

        if font["codecName"] != "" and font["isUnicodeCode"] == False:
            decoder = codecs.getincrementaldecoder(font["codecName"])(errors="ignore")
            byteCount = 0

            for a in range(len(raw)):
                character = decoder.decode(bytes([ord(raw[a])]))
                byteCount += 1

                if len(character) > 0:
                    charList.append(character)
                    widthFractionList.append(font["defaultWidthFraction"] if byteCount > 1 else font["defaultWidthFraction"] / 2)
                    codeList.append(ord(raw[a]) if byteCount == 1 else 0)

                    byteCount = 0

            return {"charList": charList, "widthFractionList": widthFractionList, "codeList": codeList}

        for a in range(0, len(raw), font["byteLength"]):
            code = ord(raw[a])

            if font["byteLength"] == 2:
                code = (ord(raw[a]) << 8) | (ord(raw[a + 1]) if a + 1 < len(raw) else 0)

            character = font["toUnicodeObject"].get(code)

            if character is None and font["byteLength"] == 1:
                character = font["encodingObject"].get(code)

            if character is None:
                character = chr(code) if font["byteLength"] == 1 or font["isUnicodeCode"] else ""

            widthFraction = font["defaultWidthFraction"]

            if font["byteLength"] == 2:
                if font["widthObject"].get(code) is not None:
                    widthFraction = font["widthObject"][code]
            elif code >= font["firstChar"] and code - font["firstChar"] < len(font["widthList"]):
                widthFraction = font["widthList"][code - font["firstChar"]] * font["widthScale"]

            charList.append(character)
            widthFractionList.append(widthFraction)
            codeList.append(code)

        return {"charList": charList, "widthFractionList": widthFractionList, "codeList": codeList}

    def _transformPoint(self, matrixList, x, y):
        return [x * matrixList[0] + y * matrixList[2] + matrixList[4], x * matrixList[1] + y * matrixList[3] + matrixList[5]]

    def _colorRgb(self, red, green, blue):
        return f"#{self._componentHex(red)}{self._componentHex(green)}{self._componentHex(blue)}"

    def _componentHex(self, value):
        clamped = max(0, min(255, math.floor(value * 255 + 0.5)))

        return f"{clamped:02x}"

    def _pathAddPoint(self, x, y):
        pointList = self._transformPoint(self.ctmList, x, y)

        if self.isPathEmpty:
            self.pathMinX = pointList[0]
            self.pathMinY = pointList[1]
            self.pathMaxX = pointList[0]
            self.pathMaxY = pointList[1]
            self.isPathEmpty = False
        else:
            self.pathMinX = min(self.pathMinX, pointList[0])
            self.pathMinY = min(self.pathMinY, pointList[1])
            self.pathMaxX = max(self.pathMaxX, pointList[0])
            self.pathMaxY = max(self.pathMaxY, pointList[1])

    def _pathPaint(self, isFill, isStroke):
        if self.isPathEmpty == False:
            self.elementList.append({
                "type": "rect" if self.isPathRectangle else "path",
                "x0": self.pathMinX,
                "y0": self.pageHeight - self.pathMaxY,
                "x1": self.pathMaxX,
                "y1": self.pageHeight - self.pathMinY,
                "color": self.fillColor if isFill else self.strokeColor,
                "isFill": isFill,
                "isStroke": isStroke
            })

        self._pathReset()

    def _pageLink(self, entryObject):
        annotsNode = self._resolve(entryObject.get("Annots"))

        if annotsNode is not None and annotsNode["kind"] == "array" and annotsNode.get("itemList") is not None:
            for a in range(len(annotsNode["itemList"])):
                annot = self._resolve(annotsNode["itemList"][a])

                if annot is not None and annot.get("entryObject") is not None:
                    subtypeNode = self._resolve(annot["entryObject"].get("Subtype"))

                    if subtypeNode is not None and subtypeNode["kind"] == "name" and subtypeNode["value"] == "Link":
                        rectNode = self._resolve(annot["entryObject"].get("Rect"))
                        actionNode = self._resolve(annot["entryObject"].get("A"))

                        uri = ""

                        if actionNode is not None and actionNode.get("entryObject") is not None:
                            uriNode = self._resolve(actionNode["entryObject"].get("URI"))

                            if uriNode is not None and (uriNode["kind"] == "string" or uriNode["kind"] == "hexString"):
                                uri = uriNode["value"]

                        if rectNode is not None and rectNode["kind"] == "array" and rectNode.get("itemList") is not None:
                            itemList = rectNode["itemList"]

                            left = self._numberValue(itemList[0] if len(itemList) > 0 else None)
                            bottom = self._numberValue(itemList[1] if len(itemList) > 1 else None)
                            right = self._numberValue(itemList[2] if len(itemList) > 2 else None)
                            top = self._numberValue(itemList[3] if len(itemList) > 3 else None)

                            self.elementList.append({
                                "type": "link",
                                "x0": min(left, right),
                                "y0": self.pageHeight - max(top, bottom),
                                "x1": max(left, right),
                                "y1": self.pageHeight - min(top, bottom),
                                "uri": uri
                            })

    def _mergeText(self, elementList):
        resultList = []

        current = None
        pendingList = []

        for a in range(len(elementList)):
            element = elementList[a]

            if element["type"] != "text":
                if current is None:
                    resultList.append(element)
                else:
                    pendingList.append(element)
            elif current is None:
                current = element
            else:
                size = current["fontSize"] if current.get("fontSize") is not None else 0
                elementSize = element["fontSize"] if element.get("fontSize") is not None else 0
                gap = element["x0"] - current["x1"]

                isSameLine = abs(element["y0"] - current["y0"]) <= size * 0.6
                isCompatibleSize = elementSize >= size * 0.45 and elementSize <= size * 1.4
                isClose = gap >= -size * 0.3 and gap <= size * 1.0

                if isSameLine and isCompatibleSize and isClose:
                    previousText = current["text"] if current.get("text") is not None else ""
                    elementText = element["text"] if element.get("text") is not None else ""
                    isSpace = gap > size * 0.15 and previousText[-1:] != " " and elementText[0:1] != " "

                    if self._wideCheck(previousText[-1:]) and self._wideCheck(elementText[0:1]):
                        isSpace = False

                    current["text"] = f"{previousText} {elementText}" if isSpace else f"{previousText}{elementText}"
                    current["x1"] = element["x1"]
                    current["y0"] = min(current["y0"], element["y0"])
                    current["y1"] = max(current["y1"], element["y1"])
                else:
                    resultList.append(current)

                    for b in range(len(pendingList)):
                        resultList.append(pendingList[b])

                    pendingList = []
                    current = element

        if current is not None:
            resultList.append(current)

        for a in range(len(pendingList)):
            resultList.append(pendingList[a])

        return resultList

    def _wideCheck(self, character):
        return character != "" and unicodedata.east_asian_width(character) in ("W", "F")

    def execute(self, pathInput):
        with open(pathInput, "rb") as file:
            self.byteList = bytes(file.read())

        self.text = self._byteText(self.byteList)
        self.position = 0

        indirectList = self._parseIndirect()

        self.indirectObject = {}

        for a in range(len(indirectList)):
            self.indirectObject[indirectList[a]["number"]] = indirectList[a]

        return self._buildPage()

    def __init__(self):
        self.delimiterSet = set(ord(value) for value in "()<>[]{}/%")
        self.whitespaceSet = set([0, 9, 10, 12, 13, 32])

        self.fontSize = 0
        self.charSpacing = 0
        self.wordSpacing = 0
        self.horizontalScale = 1
        self.leading = 0
        self.textRender = 0
        self.textRise = 0
        self.fillColor = "#000000"
        self.strokeColor = "#000000"
        self.currentFont = None
        self.pageHeight = 0
        self.elementList = []
        self.byteList = b""
        self.text = ""
        self.position = 0

        self.pathMinX = 0
        self.pathMinY = 0
        self.pathMaxX = 0
        self.pathMaxY = 0

        self.isPathEmpty = True
        self.isPathRectangle = False

        self.codecList = [
            ["UCS2", "utf-16-be"],
            ["UTF16", "utf-16-be"],
            ["RKSJ", "cp932"],
            ["GBK-EUC", "gbk"],
            ["GBpc-EUC", "gb2312"],
            ["GB-EUC", "gb2312"],
            ["KSCms-UHC", "cp949"],
            ["KSCpc-EUC", "cp949"],
            ["KSC-EUC", "euc_kr"],
            ["HKscs-B5", "big5hkscs"],
            ["ETen-B5", "big5"],
            ["B5pc", "big5"],
            ["EUC", "euc_jp"]
        ]
        self.ctmList = [1, 0, 0, 1, 0, 0]
        self.textMatrixList = [1, 0, 0, 1, 0, 0]
        self.lineMatrixList = [1, 0, 0, 1, 0, 0]
        self.graphicsStateList = []

        self.indirectObject = {}
        self.glyphObject = {
            "space": " ", "exclam": "!", "quotedbl": '"', "numbersign": "#", "dollar": "$", "percent": "%", "ampersand": "&",
            "quotesingle": "'", "parenleft": "(", "parenright": ")", "asterisk": "*", "plus": "+", "comma": ",", "hyphen": "-",
            "period": ".", "slash": "/", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
            "seven": "7", "eight": "8", "nine": "9", "colon": ":", "semicolon": ";", "less": "<", "equal": "=", "greater": ">",
            "question": "?", "at": "@", "bracketleft": "[", "backslash": "\\", "bracketright": "]", "asciicircum": "^",
            "underscore": "_", "grave": "`", "braceleft": "{", "bar": "|", "braceright": "}", "asciitilde": "~",
            "quoteleft": "\u2018", "quoteright": "\u2019", "quotedblleft": "\u201c", "quotedblright": "\u201d",
            "quotesinglbase": "\u201a", "quotedblbase": "\u201e", "endash": "\u2013", "emdash": "\u2014", "bullet": "\u2022",
            "ellipsis": "\u2026", "dagger": "\u2020", "daggerdbl": "\u2021", "perthousand": "\u2030", "fi": "\ufb01",
            "fl": "\ufb02", "degree": "\u00b0", "middot": "\u00b7", "trademark": "\u2122", "copyright": "\u00a9",
            "registered": "\u00ae", "euro": "\u20ac", "yen": "\u00a5", "sterling": "\u00a3", "section": "\u00a7",
            "paragraph": "\u00b6", "guillemotleft": "\u00ab", "guillemotright": "\u00bb", "minus": "\u2212",
            "Agrave": "\u00c0", "Aacute": "\u00c1", "Acircumflex": "\u00c2", "Atilde": "\u00c3", "Adieresis": "\u00c4", "Aring": "\u00c5",
            "AE": "\u00c6", "Ccedilla": "\u00c7", "Egrave": "\u00c8", "Eacute": "\u00c9", "Ecircumflex": "\u00ca", "Edieresis": "\u00cb",
            "Igrave": "\u00cc", "Iacute": "\u00cd", "Icircumflex": "\u00ce", "Idieresis": "\u00cf", "Eth": "\u00d0", "Ntilde": "\u00d1",
            "Ograve": "\u00d2", "Oacute": "\u00d3", "Ocircumflex": "\u00d4", "Otilde": "\u00d5", "Odieresis": "\u00d6", "multiply": "\u00d7",
            "Oslash": "\u00d8", "Ugrave": "\u00d9", "Uacute": "\u00da", "Ucircumflex": "\u00db", "Udieresis": "\u00dc", "Yacute": "\u00dd",
            "Thorn": "\u00de", "germandbls": "\u00df", "agrave": "\u00e0", "aacute": "\u00e1", "acircumflex": "\u00e2", "atilde": "\u00e3",
            "adieresis": "\u00e4", "aring": "\u00e5", "ae": "\u00e6", "ccedilla": "\u00e7", "egrave": "\u00e8", "eacute": "\u00e9",
            "ecircumflex": "\u00ea", "edieresis": "\u00eb", "igrave": "\u00ec", "iacute": "\u00ed", "icircumflex": "\u00ee",
            "idieresis": "\u00ef", "eth": "\u00f0", "ntilde": "\u00f1", "ograve": "\u00f2", "oacute": "\u00f3", "ocircumflex": "\u00f4",
            "otilde": "\u00f5", "odieresis": "\u00f6", "divide": "\u00f7", "oslash": "\u00f8", "ugrave": "\u00f9", "uacute": "\u00fa",
            "ucircumflex": "\u00fb", "udieresis": "\u00fc", "yacute": "\u00fd", "thorn": "\u00fe", "ydieresis": "\u00ff",
            "exclamdown": "\u00a1", "cent": "\u00a2", "currency": "\u00a4", "brokenbar": "\u00a6", "dieresis": "\u00a8",
            "ordfeminine": "\u00aa", "logicalnot": "\u00ac", "macron": "\u00af", "plusminus": "\u00b1", "twosuperior": "\u00b2",
            "threesuperior": "\u00b3", "acute": "\u00b4", "mu": "\u00b5", "periodcentered": "\u00b7", "cedilla": "\u00b8",
            "onesuperior": "\u00b9", "ordmasculine": "\u00ba", "onequarter": "\u00bc", "onehalf": "\u00bd", "threequarters": "\u00be",
            "questiondown": "\u00bf", "Scaron": "\u0160", "scaron": "\u0161", "Zcaron": "\u017d", "zcaron": "\u017e", "OE": "\u0152",
            "oe": "\u0153", "Ydieresis": "\u0178", "florin": "\u0192", "circumflex": "\u02c6", "tilde": "\u02dc", "dotlessi": "\u0131",
            "Lslash": "\u0141", "lslash": "\u0142", "Aogonek": "\u0104", "aogonek": "\u0105", "Cacute": "\u0106", "cacute": "\u0107",
            "Eogonek": "\u0118", "eogonek": "\u0119", "Nacute": "\u0143", "nacute": "\u0144", "Sacute": "\u015a", "sacute": "\u015b",
            "Zacute": "\u0179", "zacute": "\u017a", "Zdotaccent": "\u017b", "zdotaccent": "\u017c"
        }
