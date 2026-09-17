import sys
import os
import json
import cv2
import numpy
import urllib.request
import threading

sys.dont_write_bytecode = True

class ProcessorPassthrough(urllib.request.HTTPErrorProcessor):
    def http_response(self, request, response):
        return response

    https_response = http_response

class Test:
    def _engineCall(self, nameFile, nameOutput, password):
        payload = json.dumps({"pathInput": f"file/test/{nameFile}", "pathOutput": f"file/output/{nameOutput}/", "password": password}).encode("utf-8")

        request = urllib.request.Request(self.urlEngine, data=payload, headers={"Content-Type": "application/json"})

        with urllib.request.build_opener(ProcessorPassthrough).open(request) as response:
            return {"status": response.status, "bodyObject": json.loads(response.read().decode("utf-8"))}

    def _resultRead(self, nameOutput):
        with open(f"file/output/{nameOutput}/debug/result.json", encoding="utf-8") as file:
            return json.load(file)

    def _inkCoverage(self, nameOutput, numberPage, itemList):
        image = cv2.imread(f"file/output/{nameOutput}/debug/page/{numberPage}.jpg")

        imageInk = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        imageCovered = numpy.zeros(imageInk.shape, dtype=numpy.uint8)

        countEmpty = 0

        for a in range(len(itemList)):
            if itemList[a]["page"] != numberPage:
                continue

            bboxList = itemList[a]["bbox"]

            x0 = max(0, bboxList[0])
            y0 = max(0, bboxList[1])
            x1 = min(imageInk.shape[1], bboxList[2])
            y1 = min(imageInk.shape[0], bboxList[3])

            if x1 <= x0 or y1 <= y0:
                countEmpty += 1

                continue

            if imageInk[y0:y1, x0:x1].mean() < self.levelInkItem:
                countEmpty += 1

            imageCovered[y0:y1, x0:x1] = 1

        countInk = int(imageInk.sum())
        countInkCovered = int((imageInk * imageCovered).sum())

        return {
            "size": {"width": imageInk.shape[1], "height": imageInk.shape[0]},
            "ratio": countInkCovered / float(countInk) if countInk > 0 else 0.0,
            "countEmpty": countEmpty
        }

    def _inkDifferenceList(self, nameOutput, numberPage, itemList, isVertical):
        image = cv2.imread(f"file/output/{nameOutput}/debug/page/{numberPage}.jpg")

        imageInk = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        resultList = []

        for a in range(len(itemList)):
            if itemList[a]["page"] != numberPage:
                continue

            bboxList = itemList[a]["bbox"]

            if isVertical:
                regionList = imageInk[:, max(0, bboxList[0] - self.marginBand):bboxList[2] + self.marginBand].sum(axis=1)
                size = bboxList[3] - bboxList[1]
            else:
                regionList = imageInk[max(0, bboxList[1] - self.marginBand):bboxList[3] + self.marginBand, :].sum(axis=0)
                size = bboxList[2] - bboxList[0]

            indexList = numpy.nonzero(regionList)[0]

            if len(indexList) == 0:
                continue

            resultList.append(abs(size - (int(indexList.max()) - int(indexList.min()))))

        return resultList

    def _textPageList(self, itemList, numberPage):
        resultList = []

        for a in range(len(itemList)):
            if itemList[a]["page"] == numberPage and itemList[a]["text"].startswith(self.textTitleRotate) == False:
                resultList.append(itemList[a]["text"])

        resultList.sort()

        return resultList

    def _report(self, name, isPass, message):
        self.countTotal += 1

        if isPass:
            self.countPass += 1

        print(f"[{'PASS' if isPass else 'FAIL'}] {name} - {message}")

    def _testRotate(self):
        self._engineCall("rotate_1.pdf", "test_rotate_1", "")

        resultObject = self._resultRead("test_rotate_1")

        itemList = resultObject["itemList"]

        textPageList = []
        coverageList = []

        for a in range(self.countPageRotate):
            textPageList.append(self._textPageList(itemList, a + 1))
            coverageList.append(self._inkCoverage("test_rotate_1", a + 1, itemList))

        isTextSame = True

        for a in range(1, len(textPageList)):
            if textPageList[a] != textPageList[0]:
                isTextSame = False

        self._report("rotate/testo", isTextSame and len(textPageList[0]) == self.countLineRotate, f"righe per pagina {[len(textPageList[a]) for a in range(len(textPageList))]}, uguali su tutte le pagine: {isTextSame}")

        ratioMinimum = 1.0
        countEmpty = 0

        for a in range(len(coverageList)):
            ratioMinimum = min(ratioMinimum, coverageList[a]["ratio"])
            countEmpty += coverageList[a]["countEmpty"]

        self._report("rotate/inchiostro", ratioMinimum >= self.levelInkPage and countEmpty == 0, f"copertura minima {round(ratioMinimum, 3)}, box senza inchiostro {countEmpty}")

        isSizeSame = True

        for a in range(1, len(coverageList)):
            if coverageList[a]["size"] != coverageList[0]["size"]:
                isSizeSame = False

        self._report("rotate/dimensione", isSizeSame, f"dimensioni {[(coverageList[a]['size']['width'], coverageList[a]['size']['height']) for a in range(len(coverageList))]}")

    def _testCrop(self):
        self._engineCall("crop_1.pdf", "test_crop_1", "")

        resultObject = self._resultRead("test_crop_1")

        itemList = resultObject["itemList"]

        coverageObject = self._inkCoverage("test_crop_1", 1, itemList)

        self._report("crop/inchiostro", coverageObject["ratio"] >= self.levelInkPage and coverageObject["countEmpty"] == 0, f"copertura {round(coverageObject['ratio'], 3)}, box senza inchiostro {coverageObject['countEmpty']}")

        scale = self.sizeRasterLong / float(self.cropWidthPoint)

        widthExpected = int(round(self.cropWidthPoint * scale))
        heightExpected = int(round(self.cropHeightPoint * scale))

        isSize = abs(coverageObject["size"]["width"] - widthExpected) <= self.toleranceRaster and abs(coverageObject["size"]["height"] - heightExpected) <= self.toleranceRaster

        self._report("crop/dimensione", isSize, f"attesa {widthExpected}x{heightExpected}, trovata {coverageObject['size']['width']}x{coverageObject['size']['height']}")

        bboxTitleList = None

        for a in range(len(itemList)):
            if itemList[a]["text"].startswith(self.textTitleCrop):
                bboxTitleList = itemList[a]["bbox"]

                break

        x0Expected = int(round(self.titleX0Point * scale))
        y0Expected = int(round(self.titleY0Point * scale))

        isPosition = bboxTitleList is not None and abs(bboxTitleList[0] - x0Expected) <= self.tolerancePixel and abs(bboxTitleList[1] - y0Expected) <= self.tolerancePixel

        self._report("crop/posizione", isPosition, f"attesa ({x0Expected}, {y0Expected}), trovata {bboxTitleList[0:2] if bboxTitleList is not None else None}")

    def _markdownRead(self, nameOutput):
        with open(f"file/output/{nameOutput}/result.md", encoding="utf-8") as file:
            return file.read()

    def _testSpace(self):
        self._engineCall(self.filePdfSpace, "test_space_1", "")

        markdown = self._markdownRead("test_space_1")

        countDouble = markdown.count(self.textDouble)
        isCell = self.textCellSpace in markdown

        self._report("spazio/tabella", isCell and countDouble == 0, f"cella {self.textCellSpace!r} presente: {isCell}, doppi spazi: {countDouble}")

        self._engineCall(self.fileOfficeSpace, "test_space_2", "")

        markdown = self._markdownRead("test_space_2")

        countDouble = markdown.count(self.textDouble)
        isJoin = self.textJoinSpace in markdown
        isWrap = self.textWrapSpace in markdown

        self._report("spazio/office", isJoin and isWrap and countDouble == 0, f"continuazione: {isJoin}, riquadro: {isWrap}, doppi spazi: {countDouble}")

    def _testDate(self):
        self._engineCall(self.fileDate1904, "test_date_1", "")

        markdown1904 = self._markdownRead("test_date_1")

        self._report("data/1904", self.textDateExpected in markdown1904, f"attesa {self.textDateExpected!r}, riga {self._lineFind(markdown1904, self.textDateCell)!r}")

        self._engineCall(self.fileDate1900, "test_date_2", "")

        markdown1900 = self._markdownRead("test_date_2")

        self._report("data/1900", self.textDateExpected in markdown1900, f"attesa {self.textDateExpected!r}, riga {self._lineFind(markdown1900, self.textDateCell)!r}")

    def _lineFind(self, markdown, text):
        lineList = markdown.split("\n")

        for a in range(len(lineList)):
            if text in lineList[a]:
                return lineList[a]

        return ""

    def _testPage(self):
        resultObject = self._engineCall(self.filePage, "test_page_1", "")

        if resultObject["bodyObject"].get("response", {}).get("state") != self.stateOk:
            self._report("pagina/testo", False, f"status {resultObject['status']}, corpo {resultObject['bodyObject']}")
            self._report("pagina/inchiostro", False, "non eseguito")

            return

        itemList = self._resultRead("test_page_1")["itemList"]

        countRight = 0

        for a in range(len(self.textPageList)):
            textList = self._textPageList(itemList, a + 1)

            for b in range(len(textList)):
                if textList[b] == self.textPageList[a]:
                    countRight += 1

        self._report("pagina/testo", countRight == len(self.textPageList), f"{countRight} di {len(self.textPageList)} marcatori sulla pagina giusta")

        ratioMinimum = 1.0
        countEmpty = 0

        for a in range(len(self.textPageList)):
            coverageObject = self._inkCoverage("test_page_1", a + 1, itemList)

            ratioMinimum = min(ratioMinimum, coverageObject["ratio"])
            countEmpty += coverageObject["countEmpty"]

        self._report("pagina/inchiostro", ratioMinimum >= self.levelInkPage and countEmpty == 0, f"copertura minima {round(ratioMinimum, 3)}, box senza inchiostro {countEmpty}")

    def _testFont(self):
        self._engineCall(self.fileFont, "test_font_1", "")

        itemList = self._resultRead("test_font_1")["itemList"]

        coverageObject = self._inkCoverage("test_font_1", 1, itemList)

        self._report("font/inchiostro", coverageObject["ratio"] >= self.levelInkPage and coverageObject["countEmpty"] == 0, f"copertura {round(coverageObject['ratio'], 3)}, box senza inchiostro {coverageObject['countEmpty']}")

        countLine = 0

        for a in range(len(itemList)):
            if itemList[a]["page"] == 1:
                countLine += 1

        self._report("font/righe", countLine == self.countLineFont, f"{countLine} righe estratte, attese {self.countLineFont}")

    def _testRowEmpty(self):
        self._engineCall(self.fileRowForm, "test_row_1", "")

        countEmpty = self._rowEmptyCount(self._markdownRead("test_row_1"))

        self._report("tabella/righeModulo", countEmpty >= self.countRowForm, f"righe vuote conservate {countEmpty}, attese almeno {self.countRowForm}")

        self._engineCall(self.fileRowPhantom, "test_row_2", "")

        lineList = self._markdownRead("test_row_2").split("\n")

        indexBefore = self._lineIndex(lineList, self.textRowBefore)
        indexAfter = self._lineIndex(lineList, self.textRowAfter)

        self._report("tabella/righeFantasma", indexBefore >= 0 and indexAfter == indexBefore + 1, f"riga con {self.textRowBefore!r} alla {indexBefore}, {self.textRowAfter!r} alla {indexAfter}")

    def _rowEmptyCount(self, markdown):
        lineList = markdown.split("\n")

        result = 0

        for a in range(len(lineList)):
            text = lineList[a].strip()

            if text.startswith(self.separatorCell) == False:
                continue

            isEmpty = True

            for cell in text.strip(self.separatorCell).split(self.separatorCell):
                if cell.strip() != "" and cell.strip() != self.textCellEmpty:
                    isEmpty = False

            if isEmpty:
                result += 1

        return result

    def _lineIndex(self, lineList, text):
        for a in range(len(lineList)):
            if text in lineList[a]:
                return a

        return -1

    def _testOfficeText(self):
        self._engineCall(self.fileOfficeText, "test_office_1", "")

        markdown = self._markdownRead("test_office_1")

        countWrong = 0

        for a in range(len(self.textOfficeList)):
            if self.textOfficeList[a] not in markdown:
                countWrong += 1

        countDouble = markdown.count(self.textDouble)

        self._report("office/testo", countWrong == 0 and countDouble == 0, f"attese mancanti {countWrong} di {len(self.textOfficeList)}, doppi spazi {countDouble}")

        self._report("office/cella", self.textOfficeCell in markdown, f"riga attesa {self.textOfficeCell!r} presente: {self.textOfficeCell in markdown}")

    def _testVertical(self):
        self._engineCall(self.fileVertical, "test_vertical_1", "")

        itemList = self._resultRead("test_vertical_1")["itemList"]

        textList = self._textPageList(itemList, 1)

        isColumn = textList == sorted(self.textVerticalList)

        self._report("verticale/colonne", isColumn, f"trovate {textList}")

        coverageObject = self._inkCoverage("test_vertical_1", 1, itemList)

        self._report("verticale/inchiostro", coverageObject["ratio"] >= self.levelInkPage and coverageObject["countEmpty"] == 0, f"copertura {round(coverageObject['ratio'], 3)}, box senza inchiostro {coverageObject['countEmpty']}")

    def _testVerticalMetric(self):
        self._engineCall(self.fileVerticalMetric, "test_vertical_2", "")

        differenceList = self._inkDifferenceList("test_vertical_2", 1, self._resultRead("test_vertical_2")["itemList"], True)

        isMetric = len(differenceList) == self.countColumnMetric and max(differenceList) <= self.toleranceMetric

        self._report("verticale/metriche", isMetric, f"differenze fra riquadro e inchiostro {differenceList} px, tolleranza {self.toleranceMetric}")

    def _testVerticalCmap(self):
        self._engineCall(self.fileVerticalCmap, "test_vertical_3", "")

        differenceList = self._inkDifferenceList("test_vertical_3", 1, self._resultRead("test_vertical_3")["itemList"], True)

        isMetric = len(differenceList) == self.countColumnCmap and max(differenceList) <= self.toleranceMetric

        self._report("verticale/cmap", isMetric, f"differenze fra riquadro e inchiostro {differenceList} px, tolleranza {self.toleranceMetric}")

    def _testCmap(self):
        self._engineCall(self.fileCmap, "test_cmap_1", "")

        differenceList = self._inkDifferenceList("test_cmap_1", 1, self._resultRead("test_cmap_1")["itemList"], False)

        isMetric = len(differenceList) == self.countLineCmap and max(differenceList) <= self.toleranceMetric

        self._report("cmap/metriche", isMetric, f"differenze fra riquadro e inchiostro {differenceList} px, tolleranza {self.toleranceMetric}")

    def _testScan(self):
        self._engineCall(self.fileScan, "test_scan_1", "")

        markdown = self._markdownRead("test_scan_1")

        countWrong = 0

        for a in range(len(self.textScanList)):
            if self.textScanList[a] not in markdown:
                countWrong += 1

        countItem = len(self._resultRead("test_scan_1")["itemList"])

        self._report("scansione/testo", countWrong == 0 and countItem >= self.countItemScan, f"attese mancanti {countWrong} di {len(self.textScanList)}, item estratti {countItem}")

    def _testCrypt(self):
        self._engineCall(self.filePlain, "test_crypt_0", "")

        markdownPlain = self._markdownRead("test_crypt_0")

        for a in range(len(self.fileCryptList)):
            self._engineCall(self.fileCryptList[a][0], f"test_crypt_{a + 1}", "")

            markdown = self._markdownRead(f"test_crypt_{a + 1}")

            self._report(f"cifrato/{self.fileCryptList[a][1]}", markdown == markdownPlain, f"markdown uguale al file in chiaro: {markdown == markdownPlain}, {len(markdown)} byte")

        resultObject = self._engineCall(self.filePassword, "test_crypt_9", "")

        isRequired = resultObject["bodyObject"].get("response", {}).get("message") == self.messagePasswordRequired

        self._report("cifrato/senzaPassword", isRequired, f"corpo {resultObject['bodyObject']}")

        resultObject = self._engineCall(self.filePassword, "test_crypt_9", self.passwordWrong)

        isWrong = resultObject["bodyObject"].get("response", {}).get("message") == self.messagePasswordWrong

        self._report("cifrato/passwordErrata", isWrong, f"corpo {resultObject['bodyObject']}")

        resultObject = self._engineCall(self.filePassword, "test_crypt_9", self.passwordRight)

        if resultObject["bodyObject"].get("response", {}).get("state") != self.stateOk:
            self._report("cifrato/passwordGiusta", False, f"corpo {resultObject['bodyObject']}")

            return

        markdown = self._markdownRead("test_crypt_9")

        self._report("cifrato/passwordGiusta", markdown == markdownPlain, f"markdown uguale al file in chiaro: {markdown == markdownPlain}, {len(markdown)} byte")

    def _testBroken(self):
        resultObject = self._engineCall(self.fileBrokenImage, "test_broken_1", "")

        isImage = resultObject["status"] == 200 and resultObject["bodyObject"].get("response", {}).get("state") == self.stateError

        self._report("rotto/immagine", isImage, f"status {resultObject['status']}, corpo {resultObject['bodyObject']}")

        resultObject = self._engineCall(self.fileBrokenPdf, "test_broken_2", "")

        isPdf = resultObject["status"] == 200 and resultObject["bodyObject"].get("response", {}).get("state") == self.stateError

        self._report("rotto/pdf", isPdf, f"status {resultObject['status']}, corpo {resultObject['bodyObject']}")

    def _testExtension(self):
        resultObject = self._engineCall(self.fileExtensionWrong, "test_extension_1", "")

        isBusiness = resultObject["status"] == 200 and resultObject["bodyObject"].get("response", {}).get("state") == self.stateError

        self._report("estensione/rifiutata", isBusiness, f"status {resultObject['status']}, corpo {resultObject['bodyObject']}")

        resultObject = self._engineCall(self.fileExtensionRight, "test_extension_2", "")

        isAccepted = resultObject["status"] == 200 and resultObject["bodyObject"].get("response", {}).get("state") == self.stateOk

        self._report("estensione/accettata", isAccepted, f"status {resultObject['status']}, corpo {resultObject['bodyObject']}")

    def _testParallel(self):
        for a in range(len(self.fileParallelList)):
            self._engineCall(self.fileParallelList[a], f"test_sequence_{a + 1}", "")

        threadList = []

        for a in range(len(self.fileParallelList)):
            threadList.append(threading.Thread(target=self._engineCall, args=(self.fileParallelList[a], f"test_parallel_{a + 1}", "")))

        for a in range(len(threadList)):
            threadList[a].start()

        for a in range(len(threadList)):
            threadList[a].join()

        countSame = 0

        for a in range(len(self.fileParallelList)):
            pathSequence = f"file/output/test_sequence_{a + 1}/result.md"
            pathParallel = f"file/output/test_parallel_{a + 1}/result.md"

            if os.path.isfile(pathParallel) == False:
                continue

            with open(pathSequence, encoding="utf-8") as file:
                textSequence = file.read()

            with open(pathParallel, encoding="utf-8") as file:
                textParallel = file.read()

            if textSequence == textParallel:
                countSame += 1

        self._report("parallelo/reader", countSame == len(self.fileParallelList), f"{countSame} di {len(self.fileParallelList)} identici alla corsa sequenziale")

    def execute(self):
        self._testRotate()
        self._testCrop()
        self._testSpace()
        self._testDate()
        self._testPage()
        self._testFont()
        self._testRowEmpty()
        self._testOfficeText()
        self._testVertical()
        self._testVerticalMetric()
        self._testVerticalCmap()
        self._testCmap()
        self._testScan()
        self._testCrypt()
        self._testBroken()
        self._testExtension()
        self._testParallel()

        print(f"\n{self.countPass} di {self.countTotal} test passati")

        return 0 if self.countPass == self.countTotal else 1

    def __init__(self):
        self.urlEngine = f"{os.environ['MS_FDE_URL_API_ONNX']}/engine"

        self.levelInkItem = 0.01
        self.levelInkPage = 0.8

        self.countPageRotate = 4
        self.countLineRotate = 3

        self.textTitleRotate = "Rotation"

        self.sizeRasterLong = 1755

        self.cropWidthPoint = 495
        self.cropHeightPoint = 160

        self.textTitleCrop = "Crop box test"

        self.titleX0Point = 22
        self.titleY0Point = 22

        self.tolerancePixel = 20
        self.toleranceRaster = 2

        self.filePdfSpace = "space_1.pdf"
        self.fileOfficeSpace = "space_2.docx"

        self.textDouble = "  "
        self.textCellSpace = "| Alpha Beta |"
        self.textJoinSpace = "punteggiatura e prosegue"
        self.textWrapSpace = "Riquadro prima riga riquadro seconda riga"

        self.fileDate1904 = "date_1.xlsx"
        self.fileDate1900 = "date_2.xlsx"

        self.textDateCell = "Scadenza"
        self.textDateExpected = "2016-04-02"

        self.filePage = "page_1.pdf"

        self.textPageList = ["PAGE ONE MARKER", "PAGE TWO MARKER", "PAGE THREE MARKER"]

        self.fileFont = "font_1.pdf"

        self.countLineFont = 5

        self.fileRowForm = "jp_2.jpg"
        self.countRowForm = 9

        self.fileRowPhantom = "Wikipedia - Japan.pdf"
        self.textRowBefore = "22. Shizuoka"
        self.textRowAfter = "23. Aichi"

        self.separatorCell = "|"
        self.textCellEmpty = "\u200b"

        self.fileOfficeText = "office_1.docx"

        self.textOfficeList = [
            "Etichetta Valore della etichetta",
            "Prima riga del paragrafo\nseconda riga del paragrafo",
            "Terza riga del paragrafo\nquarta riga del paragrafo",
            "Un sistema self-contained di prova."
        ]

        self.textOfficeCell = "| cella spezzata | valore |"

        self.fileVertical = "vertical_1.pdf"

        self.fileVerticalMetric = "vertical_3.pdf"

        self.countColumnMetric = 2
        self.toleranceMetric = 20
        self.marginBand = 4

        self.fileVerticalCmap = "vertical_4.pdf"

        self.countColumnCmap = 3

        self.fileCmap = "cmap_1.pdf"

        self.countLineCmap = 3

        self.textVerticalList = ["日本語の縦書きの文書です。", "二列目は左に置かれます。", "三列目には数字の一二三があります。"]

        self.fileScan = "scan_1.pdf"

        self.textScanList = ["INVOICE", "Foundation Work", "| 3 | Concrete Material | 200 Cubics | $50 | $10,000 |"]

        self.countItemScan = 50

        self.filePlain = "crypt_0.pdf"

        self.fileCryptList = [("crypt_1.pdf", "rc4"), ("crypt_2.pdf", "aes128"), ("crypt_3.pdf", "aes256")]

        self.filePassword = "crypt_4.pdf"

        self.passwordRight = "segreto"
        self.passwordWrong = "sbagliata"

        self.messagePasswordRequired = "Password required."
        self.messagePasswordWrong = "Password wrong."

        self.fileBrokenImage = "broken_1.png"
        self.fileBrokenPdf = "broken_2.pdf"

        self.fileExtensionWrong = "crop_1.jfif"
        self.fileExtensionRight = "crop_1.pdf"

        self.stateOk = "ok"
        self.stateError = "ko"

        self.fileParallelList = ["Wikipedia - Italia.pdf", "Wikipedia - Japan.pdf"]

        self.countTotal = 0
        self.countPass = 0

sys.exit(Test().execute())
