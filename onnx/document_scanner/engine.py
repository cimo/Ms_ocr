import sys
import os
import icu
import cv2
import json
import time

sys.dont_write_bytecode = True

# Source
import layout
import table
import ocr
import image
import pdf
import office
import markdown

class Engine:
    def _extensionCategory(self, extension):
        for category in self.extensionObject:
            if extension in self.extensionObject[category]:
                return category

        return ""

    def _matchAssign(self, itemList, searchText):
        for a in range(len(itemList)):
            itemList[a]["isMatch"] = self._matchCheck(searchText, itemList[a]["text"])

    def _matchCheck(self, searchText, value):
        if searchText == "" or value == "":
            return False

        return self._matchNormalize(searchText) in self._matchNormalize(value)

    def _matchNormalize(self, text):
        result = ""

        textNormalized = icu.Normalizer2.getNFKCCasefoldInstance().normalize(text)

        for a in range(len(textNormalized)):
            if icu.Char.isUWhiteSpace(textNormalized[a]) or icu.Char.hasBinaryProperty(textNormalized[a], icu.UProperty.DEFAULT_IGNORABLE_CODE_POINT):
                continue

            result += textNormalized[a]

        return result

    def _extensionAllowed(self):
        resultObject = {"image": [], "pdf": [], "office": []}

        mimeTypeList = json.loads(os.environ["MS_O_MIME_TYPE"])

        for a in range(len(mimeTypeList)):
            extension = f".{mimeTypeList[a].split('/')[1]}"

            if mimeTypeList[a].startswith("image/"):
                resultObject["image"].append(extension)

                continue

            if mimeTypeList[a] == "application/pdf":
                resultObject["pdf"].append(extension)

                continue

            if mimeTypeList[a] == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                resultObject["office"].append(".docx")

                continue

            if mimeTypeList[a] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                resultObject["office"].append(".xlsx")

                continue

            if mimeTypeList[a] == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
                resultObject["office"].append(".pptx")

        return resultObject

    def execute(self, pathInput, pathOutput, password, searchText):
        timeStart = time.perf_counter()

        extension = os.path.splitext(pathInput)[1].lower()

        category = self._extensionCategory(extension)

        if category == "":
            return {"response": {"state": "ko", "message": "Extension not supported."}}

        for a in range(len(self.debugNameList)):
            os.makedirs(f"{pathOutput}debug/{self.debugNameList[a]}/", exist_ok=True)

        os.makedirs(f"{pathOutput}page/", exist_ok=True)

        if category == "image":
            resultObject = self.image.execute(pathInput, pathOutput)
        elif category == "pdf":
            resultObject = self.pdf.execute(pathInput, pathOutput, password)
        else:
            resultObject = self.office.execute(pathInput, pathOutput, extension)

        if "message" in resultObject:
            return {"response": {"state": "ko", "message": resultObject["message"]}}

        if resultObject["pageCount"] == 0:
            return {"response": {"state": "ko", "message": "File not readable."}}

        resultObject["markdown"] = self.markdown.execute(resultObject, extension)

        self._matchAssign(resultObject["itemList"], searchText)

        with open(f"{pathOutput}result.json", "w", encoding="utf-8") as file:
            json.dump({"layoutList": resultObject["layoutList"], "tableList": resultObject["tableList"], "itemList": resultObject["itemList"]}, file, ensure_ascii=False, indent=2)

        with open(f"{pathOutput}result.md", "w", encoding="utf-8") as file:
            file.write(resultObject["markdown"])

        timeEnd = time.perf_counter() - timeStart

        print(f"\nengine.py - Time: {round(timeEnd, 3)} - {os.path.basename(pathInput)} - Page: {resultObject['pageCount']}")

        return {"response": {"state": "ok", "message": "Task completed."}}

    def __init__(self):
        self.countThread = 0

        self.extensionObject = self._extensionAllowed()

        self.debugNameList = ["layout", "table", "ocr"]

        self.layout = layout.Layout()
        self.table = table.Table()
        self.ocr = ocr.Ocr()
        self.image = image.Image(self.layout, self.table, self.ocr)
        self.pdf = pdf.Process(self.layout, self.table, self.ocr)
        self.office = office.Office()
        self.markdown = markdown.Markdown(self.extensionObject, self.office)

        cv2.setUseOptimized(True)
        cv2.setNumThreads(self.countThread)
