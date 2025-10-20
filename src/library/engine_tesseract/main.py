import os
import logging
import ast
import cv2
import numpy

# Source
from preprocessor import main as preprocessor
from engine_craft.main import EngineCraft

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
    def _createOutputDir(self):
        os.makedirs(f"{PATH_ROOT}{PATH_FILE_OUTPUT}tesseract/{self.uniqueId}/layout/", exist_ok=True)
        #os.makedirs(f"{PATH_ROOT}{PATH_FILE_OUTPUT}tesseract/{self.uniqueId}/table/classification/", exist_ok=True)
        #os.makedirs(f"{PATH_ROOT}{PATH_FILE_OUTPUT}tesseract/{self.uniqueId}/table/wired/", exist_ok=True)
        #os.makedirs(f"{PATH_ROOT}{PATH_FILE_OUTPUT}tesseract/{self.uniqueId}/table/wireless/", exist_ok=True)
        os.makedirs(f"{PATH_ROOT}{PATH_FILE_OUTPUT}tesseract/{self.uniqueId}/export/", exist_ok=True)

    def _execute(self):
        self._createOutputDir()
    
    def __init__(self, languageValue, fileNameValue, isCuda, isDebugValue, uniqueIdValue):
        self.language = languageValue
        self.fileName = fileNameValue
        self.device = "gpu" if isCuda else "cpu"
        self.isDebug = isDebugValue
        self.uniqueId = uniqueIdValue

        EngineCraft(self.fileName, self.device, self.isDebug, self.uniqueId)

        self._execute()

        print("ok", flush=True)

#def Main(PATH_ROOT, PATH_FILE_INPUT, PATH_FILE_OUTPUT, fileName, language, isCuda, isDebug):
    #helper.executeCraft(PATH_ROOT, PATH_FILE_INPUT, PATH_FILE_OUTPUT, fileName, isCuda, isDebug)

    #imageGray, imageBox, imageResult = helper.preprocess()

    #helper.result(imageGray, imageBox, imageResult)

    #thresh, boxList = pageLayout()

# TO DO - Integrate dewarp

#python3 main.py "1_jp.jpg" "jp" "False" "True"
