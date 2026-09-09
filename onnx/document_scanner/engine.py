import sys
import os
import cv2
import time
import json
import glob
import shutil
import subprocess
import unicodedata

sys.dont_write_bytecode = True

# Source
import layout
import image
import pdf
import markdown
import table

class Processor:
    def _extensionImageAllowed(self):
        resultList = []

        mimeTypeList = json.loads(os.environ["MS_O_MIME_TYPE"])

        for a in range(len(mimeTypeList)):
            if mimeTypeList[a].startswith("image/"):
                extension = mimeTypeList[a].split("/")[1]

                resultList.append(f".{extension}")

        return resultList

    def _centerPointCalculate(self, coordinateList):
        return {
            "x": int(round((coordinateList[0] + coordinateList[2]) / 2)),
            "y": int(round((coordinateList[1] + coordinateList[3]) / 2))
        }

    def _matchCheck(self, searchText, value):
        if searchText == "" or value == "":
            return False

        text = unicodedata.normalize("NFKC", value).strip().casefold().replace(" ", "")
        textSearch = unicodedata.normalize("NFKC", searchText).strip().casefold().replace(" ", "")

        return textSearch in text

    def _layoutBuild(self, astPageList):
        resultList = []

        for a in range(len(astPageList)):
            itemList = astPageList[a]["itemMainList"] + astPageList[a]["itemSecondaryList"]

            for b in range(len(itemList)):
                resultList.append({
                    "page": astPageList[a]["number"],
                    "label": itemList[b]["label"],
                    "score": itemList[b]["score"],
                    "centerPoint": self._centerPointCalculate(itemList[b]["coordinate"])
                })

        return resultList

    def _scaleCalculate(self, astPageList, pageList):
        resultObject = {}

        astPageObject = {}

        for a in range(len(astPageList)):
            astPageObject[astPageList[a]["number"]] = astPageList[a]

        for a in range(len(pageList)):
            page = pageList[a]

            if page["number"] in astPageObject:
                astPage = astPageObject[page["number"]]

                resultObject[page["number"]] = {
                    "x": astPage["imageWidth"] / page["width"],
                    "y": astPage["imageHeight"] / page["height"]
                }

        return resultObject

    def _itemBuild(self, astPageList, pageList, searchText):
        resultList = []

        scaleObject = self._scaleCalculate(astPageList, pageList)

        for a in range(len(pageList)):
            page = pageList[a]

            if page["number"] not in scaleObject:
                continue

            scale = scaleObject[page["number"]]

            elementList = page["elementList"]

            for b in range(len(elementList)):
                if elementList[b]["type"] != "text":
                    continue

                coordinateList = [
                    elementList[b]["x0"] * scale["x"],
                    elementList[b]["y0"] * scale["y"],
                    elementList[b]["x1"] * scale["x"],
                    elementList[b]["y1"] * scale["y"]
                ]

                resultList.append({
                    "id": len(resultList) + 1,
                    "page": page["number"],
                    "centerPoint": self._centerPointCalculate(coordinateList),
                    "text": elementList[b]["text"],
                    "isMatch": self._matchCheck(searchText, elementList[b]["text"])
                })

        return resultList

    def _itemAstBuild(self, astPageList, searchText):
        resultList = []

        for a in range(len(astPageList)):
            itemList = astPageList[a]["itemMainList"] + astPageList[a]["itemSecondaryList"]

            for b in range(len(itemList)):
                if len(itemList[b]["text"]) > 0:
                    resultList.append({
                        "id": len(resultList) + 1,
                        "page": astPageList[a]["number"],
                        "centerPoint": None,
                        "text": itemList[b]["text"],
                        "isMatch": self._matchCheck(searchText, itemList[b]["text"])
                    })

        return resultList

    def _debugDrawItem(self, astPageList, pageList, pathOutput, searchText):
        scaleObject = self._scaleCalculate(astPageList, pageList)

        for a in range(len(pageList)):
            page = pageList[a]

            if page["number"] not in scaleObject:
                continue

            scale = scaleObject[page["number"]]

            imageDebug = cv2.imread(f"{pathOutput}page/{page['number']}.jpg")

            elementList = page["elementList"]

            for b in range(len(elementList)):
                if elementList[b]["type"] != "text":
                    continue

                color = (0, 200, 0) if self._matchCheck(searchText, elementList[b]["text"]) else (0, 0, 255)

                cv2.rectangle(
                    imageDebug,
                    (int(round(elementList[b]["x0"] * scale["x"])), int(round(elementList[b]["y0"] * scale["y"]))),
                    (int(round(elementList[b]["x1"] * scale["x"])), int(round(elementList[b]["y1"] * scale["y"]))),
                    color,
                    1
                )

            cv2.imwrite(f"{pathOutput}debug/engine/{page['number']}.jpg", imageDebug)

    def _segmentCollect(self, astPageList, pageList):
        resultObject = {}

        scaleObject = self._scaleCalculate(astPageList, pageList)

        for a in range(len(pageList)):
            page = pageList[a]

            if page["number"] not in scaleObject:
                continue

            scale = scaleObject[page["number"]]

            horizontalList = []
            verticalList = []

            elementList = page["elementList"]

            for b in range(len(elementList)):
                if elementList[b]["type"] != "rect" and elementList[b]["type"] != "path":
                    continue

                x0 = elementList[b]["x0"] * scale["x"]
                y0 = elementList[b]["y0"] * scale["y"]
                x1 = elementList[b]["x1"] * scale["x"]
                y1 = elementList[b]["y1"] * scale["y"]

                width = x1 - x0
                height = y1 - y0

                if height <= self.thicknessSegment and width > self.lengthSegment:
                    horizontalList.append({"position": (y0 + y1) / 2, "start": x0, "end": x1})
                elif width <= self.thicknessSegment and height > self.lengthSegment:
                    verticalList.append({"position": (x0 + x1) / 2, "start": y0, "end": y1})

            resultObject[page["number"]] = {"horizontalList": horizontalList, "verticalList": verticalList}

        return resultObject

    def _debugDrawTable(self, tableList, image, pathOutput, pageNumber):
        for a in range(len(tableList)):
            x1 = max(0, int(round(tableList[a]["coordinate"][0])) - self.marginDebugTable)
            y1 = max(0, int(round(tableList[a]["coordinate"][1])) - self.marginDebugTable)
            x2 = min(image.shape[1], int(round(tableList[a]["coordinate"][2])) + self.marginDebugTable)
            y2 = min(image.shape[0], int(round(tableList[a]["coordinate"][3])) + self.marginDebugTable)

            imageCopy = image[y1:y2, x1:x2].copy()

            cellList = tableList[a]["tableObject"]["cellList"]

            for b in range(len(cellList)):
                cv2.rectangle(
                    imageCopy,
                    (int(round(cellList[b]["coordinate"][0])) - x1, int(round(cellList[b]["coordinate"][1])) - y1),
                    (int(round(cellList[b]["coordinate"][2])) - x1, int(round(cellList[b]["coordinate"][3])) - y1),
                    self.colorDebugTableCell,
                    1
                )

            cv2.imwrite(f"{pathOutput}debug/table/{pageNumber}_{a + 1}.jpg", imageCopy)

    def _tableGridBuild(self, astPageList, pathOutput, segmentObject):
        if self.isDebug:
            if os.path.isdir(f"{pathOutput}debug/table/"):
                shutil.rmtree(f"{pathOutput}debug/table/")

            os.makedirs(f"{pathOutput}debug/table/", exist_ok=True)

        for a in range(len(astPageList)):
            itemList = astPageList[a]["itemMainList"] + astPageList[a]["itemSecondaryList"]

            tableList = []

            for b in range(len(itemList)):
                if itemList[b]["label"] == "table":
                    tableList.append(itemList[b])

            if len(tableList) == 0:
                continue

            imagePage = cv2.imread(f"{pathOutput}page/{astPageList[a]['number']}.jpg")

            segmentPage = segmentObject[astPageList[a]["number"]] if astPageList[a]["number"] in segmentObject else self.segmentEmptyObject

            for b in range(len(tableList)):
                if len(segmentObject) > 0:
                    tableList[b]["tableObject"] = self.tableVector.execute(tableList[b]["coordinate"], segmentPage)
                else:
                    tableList[b]["tableObject"] = self.tableCell.execute(tableList[b]["coordinate"], imagePage)

            if self.isDebug:
                self._debugDrawTable(tableList, imagePage, pathOutput, astPageList[a]["number"])

        if self.isDebug:
            with open(f"{pathOutput}debug/layout/ast.json", "w", encoding="utf-8") as file:
                json.dump({"pageList": astPageList}, file, ensure_ascii=False, indent=4)

    def _orphanLineList(self, elementList):
        resultList = []

        elementSortList = sorted(elementList, key=lambda element: element["y0"])

        for a in range(len(elementSortList)):
            element = elementSortList[a]

            isFound = False

            for b in range(len(resultList)):
                line = resultList[b]

                overlap = min(line["y1"], element["y1"]) - max(line["y0"], element["y0"])
                height = min(line["y1"] - line["y0"], element["y1"] - element["y0"])

                if height > 0 and overlap / height >= self.levelOrphanOverlap:
                    line["x0"] = min(line["x0"], element["x0"])
                    line["y0"] = min(line["y0"], element["y0"])
                    line["x1"] = max(line["x1"], element["x1"])
                    line["y1"] = max(line["y1"], element["y1"])
                    line["isDecorative"] = line["isDecorative"] and element["isDecorative"]

                    isFound = True

                    break

            if isFound == False:
                resultList.append({"x0": element["x0"], "y0": element["y0"], "x1": element["x1"], "y1": element["y1"], "isDecorative": element["isDecorative"]})

        resultList.sort(key=lambda line: line["y0"])

        return resultList

    def _orphanBlockList(self, lineList):
        resultList = []

        isOpen = False

        for a in range(len(lineList)):
            line = lineList[a]

            if line["isDecorative"]:
                isOpen = False

                continue

            isMerge = False

            if isOpen:
                block = resultList[len(resultList) - 1]

                gap = line["y0"] - block["y1"]
                height = line["y1"] - line["y0"]

                isOverlap = min(block["x1"], line["x1"]) > max(block["x0"], line["x0"])

                if isOverlap and gap <= height * self.levelOrphanGap:
                    block["x0"] = min(block["x0"], line["x0"])
                    block["x1"] = max(block["x1"], line["x1"])
                    block["y1"] = max(block["y1"], line["y1"])

                    isMerge = True

            if isMerge == False:
                resultList.append({"x0": line["x0"], "y0": line["y0"], "x1": line["x1"], "y1": line["y1"]})

                isOpen = True

        return resultList

    def _orphanBuild(self, astPageList, pageList):
        scaleObject = self._scaleCalculate(astPageList, pageList)

        astPageObject = {}

        for a in range(len(astPageList)):
            astPageObject[astPageList[a]["number"]] = astPageList[a]

        for a in range(len(pageList)):
            page = pageList[a]

            if page["number"] not in scaleObject:
                continue

            scale = scaleObject[page["number"]]
            astPage = astPageObject[page["number"]]

            itemList = astPage["itemMainList"] + astPage["itemSecondaryList"]

            orphanList = []

            for b in range(len(page["elementList"])):
                element = page["elementList"][b]

                if element["type"] != "text":
                    continue

                x0 = element["x0"] * scale["x"]
                y0 = element["y0"] * scale["y"]
                x1 = element["x1"] * scale["x"]
                y1 = element["y1"] * scale["y"]

                centerX = (x0 + x1) / 2
                centerY = (y0 + y1) / 2

                isInside = False

                for c in range(len(itemList)):
                    coordinate = itemList[c]["coordinate"]

                    if centerX >= coordinate[0] and centerX <= coordinate[2] and centerY >= coordinate[1] and centerY <= coordinate[3]:
                        isInside = True

                        break

                if isInside == False:
                    orphanList.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "isDecorative": layout.textDecorativeCheck(element["text"])})

            if len(orphanList) == 0:
                continue

            blockList = self._orphanBlockList(self._orphanLineList(orphanList))

            for b in range(len(blockList)):
                block = blockList[b]

                astPage["itemMainList"].append({
                    "label": "text",
                    "score": 0.0,
                    "coordinate": [block["x0"], block["y0"], block["x1"], block["y1"]],
                    "boxList": [
                        [int(round(block["x0"])), int(round(block["y0"]))],
                        [int(round(block["x1"])), int(round(block["y0"]))],
                        [int(round(block["x1"])), int(round(block["y1"]))],
                        [int(round(block["x0"])), int(round(block["y1"]))]
                    ],
                    "flow": "main",
                    "order": 0
                })

            astPage["itemMainList"] = self.layoutImage._orderArrange(0, astPage["itemMainList"], astPage["imageWidth"], astPage["imageHeight"])

            for b in range(len(astPage["itemMainList"])):
                astPage["itemMainList"][b]["order"] = b + 1

    def _pageImageGenerate(self, mode, pathOutput, pathInput):
        pathPage = f"{pathOutput}page/"

        if os.path.isdir(pathPage):
            shutil.rmtree(pathPage)

        os.makedirs(pathPage, exist_ok=True)

        if mode == "single":
            cv2.imwrite(f"{pathPage}1.jpg", cv2.imread(pathInput))
        elif mode == "multiple":
            subprocess.run(["pdftoppm", "-jpeg", "-r", "150", pathInput, f"{pathPage}page"], capture_output=True, text=True)

            fileNameList = glob.glob(f"{pathPage}page-*.jpg")

            for a in range(len(fileNameList)):
                pageNumber = int(os.path.splitext(os.path.basename(fileNameList[a]))[0].split("-")[1])

                os.rename(fileNameList[a], f"{pathPage}{pageNumber}.jpg")

    def _astBuild(self, extension, pathOutput, pathInput, fileName):
        resultObject = {"pageList": []}

        if extension in self.extensionImageList:
            self._pageImageGenerate("single", pathOutput, pathInput)

            resultObject = self.layoutImage.execute(pathOutput, fileName, f"{pathOutput}page/")
        elif extension == ".pdf":
            self._pageImageGenerate("multiple", pathOutput, pathInput)

            resultObject = self.layoutImage.execute(pathOutput, fileName, f"{pathOutput}page/")
        elif extension == ".docx":
            resultObject = self.layoutOfficeDocx.execute(pathInput, pathOutput, fileName)
        elif extension == ".xlsx":
            resultObject = self.layoutOfficeXlsx.execute(pathInput, pathOutput, fileName)
        elif extension == ".pptx":
            resultObject = self.layoutOfficePptx.execute(pathInput, pathOutput, fileName)

        return resultObject["pageList"]

    def execute(self, pathOutput, searchText, pathInput, fileName):
        timeStart = time.perf_counter()

        extension = os.path.splitext(pathInput)[1].lower()

        astPageList = self._astBuild(extension, pathOutput, pathInput, fileName)

        markdownText = ""
        pageCount = 0

        pageList = []

        isLayoutImage = False

        if extension in self.extensionImageList:
            isLayoutImage = True

            self._tableGridBuild(astPageList, pathOutput, {})

            pageList = self.imageReader.execute(f"{pathOutput}page/", astPageList)

            self._orphanBuild(astPageList, pageList)

            markdownPage = markdown.Page()
            markdownText = markdownPage.execute(astPageList, pageList)

            pageCount = len(pageList)
        elif extension == ".pdf":
            isLayoutImage = True

            pdfReader = pdf.Reader()
            pageList = pdfReader.execute(pathInput)

            self._tableGridBuild(astPageList, pathOutput, self._segmentCollect(astPageList, pageList))

            self._orphanBuild(astPageList, pageList)

            markdownPage = markdown.Page()
            markdownText = markdownPage.execute(astPageList, pageList)

            pageCount = len(pageList)
        elif extension == ".docx":
            markdownDocx = markdown.Docx()
            markdownText = markdownDocx.execute(astPageList)

            pageCount = len(astPageList)
        elif extension == ".xlsx":
            markdownXlsx = markdown.Xlsx()
            markdownText = markdownXlsx.execute(astPageList)

            pageCount = len(astPageList)
        elif extension == ".pptx":
            markdownPptx = markdown.Pptx()
            markdownText = markdownPptx.execute(astPageList)

            pageCount = len(astPageList)

        layoutList = []
        itemList = []

        if isLayoutImage:
            layoutList = self._layoutBuild(astPageList)
            itemList = self._itemBuild(astPageList, pageList, searchText)
        else:
            itemList = self._itemAstBuild(astPageList, searchText)

        os.makedirs(pathOutput, exist_ok=True)

        with open(f"{pathOutput}{self.markdownFileName}", "w", encoding="utf-8", errors="replace") as file:
            file.write(markdownText)

        with open(f"{pathOutput}{self.resultFileName}", "w", encoding="utf-8") as file:
            json.dump({"layoutList": layoutList, "itemList": itemList}, file, ensure_ascii=False, indent=2)

        if self.isDebug:
            if os.path.isdir(f"{pathOutput}debug/engine/"):
                shutil.rmtree(f"{pathOutput}debug/engine/")

            os.makedirs(f"{pathOutput}debug/engine/", exist_ok=True)

            self._debugDrawItem(astPageList, pageList, pathOutput, searchText)

        timeEnd = time.perf_counter() - timeStart

        print(f"\nEngine.py - Time: {round(timeEnd, 3)} - {fileName} - Page: {pageCount}")

        resultObject = {"pageCount": pageCount, "layoutList": layoutList, "itemList": itemList}

        return resultObject

    def __init__(self):
        self.isDebug = os.environ["MS_O_IS_DEBUG"] == "true"

        self.thicknessSegment = 3.0
        self.lengthSegment = 8.0

        self.marginDebugTable = 10
        self.colorDebugTableCell = (0, 0, 255)

        self.segmentEmptyObject = {"horizontalList": [], "verticalList": []}

        self.levelOrphanOverlap = 0.5
        self.levelOrphanGap = 1.5

        self.tableCell = table.Cell()
        self.tableVector = table.Vector()

        self.layoutImage = layout.Image()
        self.layoutOfficeDocx = layout.Office.Docx()
        self.layoutOfficeXlsx = layout.Office.Xlsx()
        self.layoutOfficePptx = layout.Office.Pptx()

        self.resultFileName = "result.json"
        self.markdownFileName = "result.md"

        self.extensionImageList = self._extensionImageAllowed()

        self.imageReader = image.Reader()
