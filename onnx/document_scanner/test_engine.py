import sys
import os
import unicodedata
import cv2
import json
import time
import shutil

sys.dont_write_bytecode = True

# Source
import test_image
import test_office
import test_markdown

class Engine:
    def _matchCheck(self, searchText, value):
        if searchText == "" or value == "":
            return False

        text = unicodedata.normalize("NFKC", value).strip().casefold().replace(" ", "")
        textSearch = unicodedata.normalize("NFKC", searchText).strip().casefold().replace(" ", "")

        return textSearch in text

    def _matchAssign(self, itemList, searchText):
        for a in range(len(itemList)):
            itemList[a]["isMatch"] = self._matchCheck(searchText, itemList[a]["text"])

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

    def execute(self, pathInput, pathOutput, searchText):
        timeStart = time.perf_counter()

        for a in range(len(self.debugNameList)):
            os.makedirs(f"{pathOutput}debug/{self.debugNameList[a]}/", exist_ok=True)

        os.makedirs(f"{pathOutput}page/", exist_ok=True)

        extension = os.path.splitext(pathInput)[1].lower()

        if extension in self.extensionObject["image"]:
            resultObject = self.image.execute(pathInput, pathOutput)

            resultObject["markdown"] = self.markdownImage.execute(resultObject["layoutList"], resultObject["tableList"], resultObject["itemList"])
        elif extension in self.extensionObject["pdf"]:
            # to do
            return
        elif extension in self.extensionObject["office"]:
            resultObject = self.office.execute(pathInput, pathOutput, extension)

            resultObject["markdown"] = self.markdownOffice.execute(resultObject["astPageList"], resultObject["tableList"], extension)

        self._matchAssign(resultObject["itemList"], searchText)

        with open(f"{pathOutput}result.json", "w", encoding="utf-8") as file:
            json.dump({"layoutList": resultObject["layoutList"], "tableList": resultObject["tableList"], "itemList": resultObject["itemList"]}, file, ensure_ascii=False, indent=2)

        with open(f"{pathOutput}result.md", "w", encoding="utf-8") as file:
            file.write(resultObject["markdown"])

        timeEnd = time.perf_counter() - timeStart

        print(f"\ntest_engine.py - Time: {round(timeEnd, 3)} - {os.path.basename(pathInput)} - Page: {resultObject['pageCount']}")

        return {"response": {"state": "ok", "message": "Task completed."}}

    def __init__(self):
        self.countThread = 0

        self.extensionObject = self._extensionAllowed()

        self.debugNameList = ["layout", "table", "ocr"]

        self.image = test_image.Image()
        self.markdownImage = test_markdown.Markdown.Image()

        # to do pdf

        self.office = test_office.Office()
        self.markdownOffice = test_markdown.Markdown.Office()

        cv2.setUseOptimized(True)
        cv2.setNumThreads(self.countThread)
