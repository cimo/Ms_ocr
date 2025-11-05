import os
import logging
import ast
import json
import subprocess

# Source
from craft_detection.main import CraftDetection
from image_processor import main as imageProcessor

def _checkEnvVariable(varKey):
    if os.environ.get(varKey) is None:
        logging.exception("Environment variable %s do not exist", varKey)
    else:
        if os.environ.get(varKey).lower() == "true":
            return True
        if os.environ.get(varKey).lower() == "false":
            return False
        if os.environ.get(varKey).isnumeric():
            return int(os.environ.get(varKey))
        if os.environ.get(varKey).startswith("[") and os.environ.get(varKey).endswith("]"):
            return ast.literal_eval(os.environ.get(varKey))

    return os.environ.get(varKey)

ENV_NAME = _checkEnvVariable("ENV_NAME")
PATH_ROOT = _checkEnvVariable("PATH_ROOT")
PATH_FILE_INPUT = _checkEnvVariable("MS_O_PATH_FILE_INPUT")
PATH_FILE_OUTPUT = _checkEnvVariable("MS_O_PATH_FILE_OUTPUT")

class EngineTesseract:
    def _subprocess(self, index):
        resultLanguage = ""
        resultPsm = 6

        if self.language == "en":
            resultLanguage = "eng"
        elif self.language == "jp":
            resultLanguage = "jpn"
        elif self.language == "jp_vert":
            resultLanguage = "jpn_vert"
            resultPsm = 5

        subprocess.run([
            f"{PATH_ROOT}src/library/engine_tesseract/executable",
            f"{PATH_ROOT}{PATH_FILE_OUTPUT}tesseract/{self.uniqueId}/layout/{index}_crop.jpg",
            f"{PATH_ROOT}{PATH_FILE_OUTPUT}tesseract/{self.uniqueId}/export/{self.fileNameSplit}",
            f"-l", f"eng+{resultLanguage}" if resultLanguage != "eng" else resultLanguage,
            "--psm", str(resultPsm),
            "--oem", "1",
            "-c", "preserve_interword_spaces=0",
            "-c", "textord_tabfind_vertical_text=0",
            "-c", "textord_min_xheight=10",
            "-c", "classify_enable_learning=1",
            "-c", "classify_enable_adaptive_matcher=1",
            "-c", "classify_use_pre_adapted_templates=0",
            "-c", "classify_bln_numeric_mode=1",
            "-c", "enable_noise_removal=1",
            "-c", "noise_maxperword=16",
            "-c", "noise_maxperblob=16",
            "-c", "tessedit_create_txt=0",
            "-c", "tessedit_create_hocr=0",
            "-c", "tessedit_create_alto=0",
            "-c", "tessedit_create_page_xml=0",
            "-c", "tessedit_create_lstmbox=0",
            "-c", "tessedit_create_tsv=0",
            "-c", "tessedit_create_wordstrbox=0",
            "-c", "tessedit_create_pdf=0",
            "-c", "tessedit_create_boxfile=0"
        ], check=True)

    def _inference(self, resultMainList, imageOpen):
        resultList = []

        pilImage, pilImageDraw = imageProcessor.pilImage(imageOpen)

        for index, resultMain in enumerate(resultMainList):
            bboxList = resultMain["bbox_list"]

            left = int(bboxList[0])
            top = int(bboxList[1])
            width = int(bboxList[2])
            height = int(bboxList[3])
            right = left + width
            bottom = top + height

            imageCrop = imageOpen[top:bottom, left:right]

            resizeMultiple = imageProcessor.resizeMultiple(imageCrop)

            imageProcessor.write(f"{PATH_ROOT}{PATH_FILE_OUTPUT}tesseract/{self.uniqueId}/layout/{index}.jpg", "_crop", resizeMultiple["result"])

            self._subprocess(index)

            text = ""

            with open(f"{PATH_ROOT}{PATH_FILE_OUTPUT}tesseract/{self.uniqueId}/export/{self.fileNameSplit}.txt", "r", encoding="utf-8") as file:
                text = file.read().strip()

                pilFont = imageProcessor.pilFont(text, width, height, self.fontName)
                pilImageDraw.text((left, top), text, font=pilFont, fill=(0, 0, 0))
        
            resultList.append({
                "bbox_list": bboxList,
                "text": text
            })
        
        pilImage.save(f"{PATH_ROOT}{PATH_FILE_OUTPUT}tesseract/{self.uniqueId}/export/{self.fileNameSplit}_result.pdf", format="PDF")

        with open(f"{PATH_ROOT}{PATH_FILE_OUTPUT}tesseract/{self.uniqueId}/export/{self.fileNameSplit}_result.json", "w", encoding="utf-8") as file:
            json.dump(resultList, file, ensure_ascii=False, indent=2)

    def _createOutputDir(self):
        os.makedirs(f"{PATH_ROOT}{PATH_FILE_OUTPUT}tesseract/{self.uniqueId}/layout/", exist_ok=True)
        os.makedirs(f"{PATH_ROOT}{PATH_FILE_OUTPUT}tesseract/{self.uniqueId}/export/", exist_ok=True)

    def _execute(self, languageValue, fileNameValue, uniqueIdValue):
        self.language = languageValue
        self.fileName = fileNameValue
        self.fileNameSplit = ".".join(self.fileName.split(".")[:-1])
        self.uniqueId = uniqueIdValue
        
        craftDetection = CraftDetection(self.fileName, self.uniqueId)

        self._createOutputDir()

        imageOpen, _, _ = imageProcessor.open(f"{PATH_ROOT}{PATH_FILE_INPUT}{self.fileName}")
        
        self._inference(craftDetection.resultMainList, imageOpen)

        print("ok", flush=True)

    def __init__(self):
        os.environ["TESSDATA_PREFIX"] = f"{PATH_ROOT}src/library/engine_tesseract/language/"

        self.language = ""
        self.fileName = ""
        self.uniqueId = ""
        self.fontName = "NotoSansJP-Regular.ttf"
