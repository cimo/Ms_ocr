import sys
import subprocess
import cv2
import numpy

sys.dont_write_bytecode = True

# Source
import pdf_parser
from helper import boxFromPointList, centerPointCalculate, imageInkBuild, boxDebugWrite, astWrite

class Raster:
    def astBuild(self, pathOutput, pageObject):
        astPage = self.layout.execute(pathOutput, pageObject["image"], pageObject["number"])

        return {"astPage": astPage, "tablePageList": self.table.execute(astPage, pageObject["image"])}

    def pageAssemble(self, astPage, itemPageList, tablePageList, pageObject, pathOutput, countTable):
        self.layout.itemOrder(astPage, itemPageList)

        self.table.orderAssign(astPage, tablePageList)

        self.layout.mediaWrite(astPage, pageObject["image"], pathOutput)

        self.table.cellRefine(tablePageList, itemPageList, astPage["direction"])

        self.table.textAssign(tablePageList, itemPageList, astPage["direction"])

        self.table.debugWrite(tablePageList, pageObject["image"], itemPageList, pathOutput, pageObject["number"], countTable)

        return self.table.resultBuild(tablePageList, countTable, pageObject["number"])

    def documentBuild(self, astPageList, pathOutput, countPage, tableList, itemList):
        self.layout.flowAssign(astPageList)

        layoutList = []

        for a in range(len(astPageList)):
            layoutList = layoutList + self.layout.resultBuild(astPageList[a], len(layoutList))

        if self.isDebug:
            astWrite(pathOutput, astPageList)

        return {
            "pageCount": countPage,
            "directionList": self.layout.directionBuild(astPageList),
            "layoutList": layoutList,
            "tableList": tableList,
            "itemList": itemList
        }

    def __init__(self, isDebug, layout, table, ocr):
        self.isDebug = isDebug

        self.layout = layout
        self.table = table
        self.ocr = ocr

        self.image = Raster.Image(self)
        self.pdf = Raster.Pdf(self)

    class Image:
        def _pageBuild(self, pathInput, pathOutput):
            image = cv2.imread(pathInput)

            if image is None:
                return []

            if self.raster.isDebug:
                cv2.imwrite(f"{pathOutput}debug/page/{self.numberPageFirst}.jpg", image)

            return [{"number": self.numberPageFirst, "image": image}]

        def execute(self, pathInput, pathOutput):
            pageList = self._pageBuild(pathInput, pathOutput)

            astPageList = []
            tableList = []
            itemList = []

            for a in range(len(pageList)):
                astObject = self.raster.astBuild(pathOutput, pageList[a])

                astPageList.append(astObject["astPage"])

                itemPageList = self.raster.ocr.execute(pageList[a]["image"], astObject["tablePageList"], len(itemList), pageList[a]["number"], pathOutput)

                tableList = tableList + self.raster.pageAssemble(astObject["astPage"], itemPageList, astObject["tablePageList"], pageList[a], pathOutput, len(tableList))
                itemList = itemList + itemPageList

            return self.raster.documentBuild(astPageList, pathOutput, len(pageList), tableList, itemList)

        def __init__(self, raster):
            self.numberPageFirst = 1

            self.raster = raster

    class Pdf:
        def _pageBuild(self, password, pathInput, pathOutput):
            argumentList = ["pdftoppm", "-jpeg", "-cropbox", "-scale-to", "1755"]

            if len(password) > 0:
                argumentList = argumentList + ["-upw", password, "-opw", password]

            runObject = subprocess.run(argumentList + [pathInput], capture_output=True)

            if runObject.returncode != 0:
                print(f"raster.py - pdftoppm - {runObject.stderr.decode('utf-8', 'replace').strip()}")

                return []

            byteFileList = self._pageStreamSplit(runObject.stdout)

            resultList = []

            for a in range(len(byteFileList)):
                numberPage = a + 1

                if self.raster.isDebug:
                    with open(f"{pathOutput}debug/page/{numberPage}.jpg", "wb") as file:
                        file.write(byteFileList[a])

                resultList.append({"number": numberPage, "image": cv2.imdecode(numpy.frombuffer(byteFileList[a], dtype=numpy.uint8), cv2.IMREAD_COLOR)})

            return resultList

        def _pageStreamSplit(self, byteList):
            markerStart = b"\xff\xd8\xff"

            resultList = []

            position = byteList.find(markerStart)

            while position >= 0:
                positionNext = byteList.find(markerStart, position + len(markerStart))

                resultList.append(byteList[position:positionNext] if positionNext >= 0 else byteList[position:])

                position = positionNext

            return resultList

        def _itemBuild(self, pageReader, image, pdfParser, countStart, numberPage):
            if pageReader is None or pageReader["width"] == 0 or pageReader["height"] == 0:
                return []

            imageHeight, imageWidth = image.shape[0:2]

            scaleX = imageWidth / pageReader["width"]
            scaleY = imageHeight / pageReader["height"]

            elementList = pdfParser.mergeText(self._widthCorrect(pageReader["elementList"], image, pageReader["rotate"], scaleX, scaleY))

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
                    "centerPoint": centerPointCalculate(bboxList),
                    "text": elementList[a]["text"],
                    "isMatch": False
                })

            return resultList

        def _widthCorrect(self, elementList, image, rotate, scaleX, scaleY):
            isRotate = rotate == 90 or rotate == 270

            elementEstimatedList = []

            for a in range(len(elementList)):
                if elementList[a]["type"] != "text" or elementList[a]["isWidthEstimated"] == False:
                    continue

                elementEstimatedList.append({"element": elementList[a], "isVertical": elementList[a]["isVertical"] != isRotate})

            if len(elementEstimatedList) == 0:
                return elementList

            imageInk = imageInkBuild(image)

            boxList = self._boxCollect(image, imageInk)

            for a in range(len(boxList)):
                for b in range(len(self.axisList)):
                    isVertical = self.axisList[b]

                    nameFlow = "y" if isVertical else "x"
                    nameCross = "x" if isVertical else "y"

                    scaleFlow = scaleY if isVertical else scaleX
                    scaleCross = scaleX if isVertical else scaleY

                    groupList = self._groupCollect(elementEstimatedList, boxList[a], isVertical, nameFlow, nameCross, scaleFlow, scaleCross)

                    if len(groupList) == 0:
                        continue

                    flowStart = groupList[0][f"{nameFlow}0"]
                    flowEnd = groupList[len(groupList) - 1][f"{nameFlow}1"]

                    edge = self._inkEdge(imageInk, boxList[a], isVertical) / scaleFlow

                    if flowEnd <= flowStart or edge <= flowStart:
                        continue

                    ratio = (edge - flowStart) / (flowEnd - flowStart)

                    for c in range(len(groupList)):
                        groupList[c][f"{nameFlow}1"] = groupList[c][f"{nameFlow}0"] + (groupList[c][f"{nameFlow}1"] - groupList[c][f"{nameFlow}0"]) * ratio

            return elementList

        def _groupCollect(self, elementEstimatedList, boxObject, isVertical, nameFlow, nameCross, scaleFlow, scaleCross):
            resultList = []

            for a in range(len(elementEstimatedList)):
                if elementEstimatedList[a]["isVertical"] != isVertical:
                    continue

                element = elementEstimatedList[a]["element"]

                cross1 = max(element[f"{nameCross}0"] * scaleCross, boxObject[f"{nameCross}0"])
                cross2 = min(element[f"{nameCross}1"] * scaleCross, boxObject[f"{nameCross}1"])

                if cross2 - cross1 < (element[f"{nameCross}1"] - element[f"{nameCross}0"]) * scaleCross * self.levelBoxOverlap:
                    continue

                flowStart = element[f"{nameFlow}0"] * scaleFlow

                if flowStart < boxObject[f"{nameFlow}0"] or flowStart > boxObject[f"{nameFlow}1"]:
                    continue

                resultList.append(element)

            resultList.sort(key=lambda elementObject: elementObject[f"{nameFlow}0"])

            return resultList

        def _boxCollect(self, image, imageInk):
            detectionList = self.raster.ocr.boxDetect(image)

            resultList = []

            for a in range(len(detectionList)):
                bboxList = boxFromPointList(detectionList[a]["coordinate"])

                x0 = max(0, bboxList[0])
                y0 = max(0, bboxList[1])
                x1 = min(imageInk.shape[1], bboxList[2])
                y1 = min(imageInk.shape[0], bboxList[3])

                if x1 <= x0 or y1 <= y0:
                    continue

                resultList.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1})

            return resultList

        def _inkEdge(self, imageInk, boxObject, isVertical):
            region = imageInk[boxObject["y0"]:boxObject["y1"], boxObject["x0"]:boxObject["x1"]]

            valueList = region.sum(axis=1) if isVertical else region.sum(axis=0)

            start = boxObject["y0"] if isVertical else boxObject["x0"]

            for a in range(len(valueList) - 1, -1, -1):
                if valueList[a] > 0:
                    return start + a + 1

            return start

        def _debugWrite(self, itemList, image, pathOutput, numberPage):
            if self.raster.isDebug == False:
                return

            bboxList = []

            for a in range(len(itemList)):
                bboxList.append(itemList[a]["bbox"])

            boxDebugWrite(image, bboxList, f"{pathOutput}debug/ocr/{numberPage}.jpg")

        def _pageReaderCollect(self, pageReaderList, countPage):
            if len(pageReaderList) != countPage:
                print(f"raster.py - page count - raster {countPage}, text {len(pageReaderList)}")

            resultObject = {}

            for a in range(len(pageReaderList)):
                resultObject[pageReaderList[a]["number"]] = pageReaderList[a]

            return resultObject

        def execute(self, pathInput, pathOutput, password):
            pageList = self._pageBuild(password, pathInput, pathOutput)

            pdfParser = pdf_parser.PdfParser()

            parserObject = pdfParser.execute(pathInput, password)

            if parserObject["message"] != "":
                return {"message": parserObject["message"]}

            pageReaderObject = self._pageReaderCollect(parserObject["pageList"], len(pageList))

            astPageList = []
            tableList = []
            itemList = []

            for a in range(len(pageList)):
                astObject = self.raster.astBuild(pathOutput, pageList[a])

                astPageList.append(astObject["astPage"])

                numberPage = pageList[a]["number"]

                itemPageList = self._itemBuild(pageReaderObject[numberPage] if numberPage in pageReaderObject else None, pageList[a]["image"], pdfParser, len(itemList), numberPage)

                if len(itemPageList) == 0:
                    itemPageList = self.raster.ocr.execute(pageList[a]["image"], astObject["tablePageList"], len(itemList), numberPage, pathOutput)
                else:
                    self._debugWrite(itemPageList, pageList[a]["image"], pathOutput, numberPage)

                tableList = tableList + self.raster.pageAssemble(astObject["astPage"], itemPageList, astObject["tablePageList"], pageList[a], pathOutput, len(tableList))
                itemList = itemList + itemPageList

            return self.raster.documentBuild(astPageList, pathOutput, len(pageList), tableList, itemList)

        def __init__(self, raster):
            self.levelBoxOverlap = 0.5

            self.axisList = [False, True]

            self.raster = raster
