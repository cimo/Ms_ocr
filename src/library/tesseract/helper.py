import sys
import os
import subprocess
import logging
import ast
import numpy
import cv2

# Source
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessor import helper as preprocessorHelper

def checkEnvVariable(varKey):
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

ENV_NAME = checkEnvVariable("ENV_NAME")
PATH_ROOT = checkEnvVariable("PATH_ROOT")
PATH_FILE_INPUT = checkEnvVariable("MS_O_PATH_FILE_INPUT")
PATH_FILE_OUTPUT = checkEnvVariable("MS_O_PATH_FILE_OUTPUT")

def locationFromEnvName():
    result = ENV_NAME.split("_")[-1]

    if result == "local":
        result = "jp"

    return result

fileName = sys.argv[1]
language = sys.argv[2]
isCuda = sys.argv[3]
isDebug = sys.argv[4]

sizeMax = 2048
ratioMultiplier = 4.0

def _addBorder(image, color, unit):
    return cv2.rectangle(image, (0, image.shape[0]), (image.shape[1], 0), color, unit)

def _backgroundCheck(image):
    imageBgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    imageHsv = cv2.cvtColor(imageBgr, cv2.COLOR_BGR2HSV)

    colorMin = numpy.array([0, 0, 0])
    colorMax = numpy.array([179, 255, 146])
    mask = cv2.inRange(imageHsv, colorMin, colorMax)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilate = cv2.dilate(mask, kernel, iterations=5)

    return 255 - cv2.bitwise_and(dilate, mask)

def _binarization(isInverted, image, blur, unitThresholdA, unitThresholdB):
    thresholdBinary = cv2.THRESH_BINARY

    if isInverted:
        thresholdBinary = cv2.THRESH_BINARY_INV

    imageBlur = cv2.GaussianBlur(image, (blur, blur), 0)
    imageDivide = cv2.divide(image, imageBlur, scale=255)

    return cv2.adaptiveThreshold(
        imageDivide,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdBinary,
        unitThresholdA,
        unitThresholdB,
    )

def _noiseRemove(image, unit):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (unit, unit))

    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def _cropFix(image):
    if image is None or image.size == 0:
        print("Error: Crop not generated!")

        return None
        
    imageBorder = _addBorder(image, (255, 255, 255), 2)

    backgroundCheckResult = _backgroundCheck(image)

    pixelQuantity = round((backgroundCheckResult > 0).mean(), 1)

    if pixelQuantity == 1.0 or pixelQuantity <= 0.4:
        imageBinarization = _binarization(True, imageBorder, 91, 255, 0)

        imageResult = _addBorder(imageBinarization, (255, 255, 255), 3)
    else:
        imageResult = _binarization(False, imageBorder, 91, 255, 5)

    return _noiseRemove(imageResult, 2)

def _subprocess(fileNameSplit):
    resultLanguage = ""
    resultPsm = 6

    if language == "en":
        resultLanguage = "eng"
    elif language == "jp":
        resultLanguage = "jpn"
    elif language == "jp_vert":
        resultLanguage = "jpn_vert"
        resultPsm = 5

    os.environ["TESSDATA_PREFIX"] = f"{PATH_ROOT}src/library/tesseract/language/"
    
    subprocess.run([
        f"{PATH_ROOT}src/library/tesseract/executable",
        f"{PATH_ROOT}{PATH_FILE_OUTPUT}tesseract/{fileNameSplit}_crop.png",
        f"{PATH_ROOT}{PATH_FILE_OUTPUT}tesseract/{fileNameSplit}",
        f"-l", resultLanguage,
        "--oem", "1",
        "--psm", str(resultPsm),
        "-c", "preserve_interword_spaces=1",
        #"-c", "page_separator=''",
        #"-c", "tessedit_char_blacklist=''",
        #"-c", "tessedit_create_txt=1",
        #"-c", "tessedit_create_hocr=1",
        #"-c", "tessedit_create_alto=1",
        #"-c", "tessedit_create_page_xml=1",
        #"-c", "tessedit_create_lstmbox=1",
        #"-c", "tessedit_create_tsv=1",
        #"-c", "tessedit_create_wordstrbox=1",
        #"-c", "tessedit_create_pdf=1",
        #"-c", "tessedit_create_boxfile=1"
    ], check=True)

def executeCraft():
    subprocess.run([
        "python3",
        f"{PATH_ROOT}src/library/craft/main.py",
        PATH_ROOT,
        PATH_FILE_INPUT,
        f"{PATH_FILE_OUTPUT}craft/",
        fileName,
        isCuda,
        isDebug
    ], check=True)

def preprocess():
    print(f"Load file: {PATH_ROOT}{PATH_FILE_INPUT}{fileName}\r")

    imageRead = preprocessorHelper.read(f"{PATH_ROOT}{PATH_FILE_INPUT}{fileName}")

    _, _, ratio, imageResize, _ = preprocessorHelper.resize(imageRead, sizeMax)

    imageGray = preprocessorHelper.gray(imageResize)

    imageResult = imageGray.copy()
    imageResult.fill(255)

    return ratio, imageGray, imageResult

def result(ratio, imageGray, imageResult):
    fileNameSplit, fileExtensionSplit = os.path.splitext(fileName)

    with open(f"{PATH_ROOT}{PATH_FILE_OUTPUT}craft/{fileNameSplit}.txt", "r") as file:
        lineList = file.readlines()

    for line in lineList:
        coordinateList = list(map(int, line.strip().split(",")))
        boxList = numpy.array(coordinateList).reshape((-1, 2))

        left = int(boxList[0][0] * ratio)
        top = int(boxList[0][1] * ratio)
        right = int(boxList[2][0] * ratio)
        bottom = int(boxList[2][1] * ratio)

        imageCrop = imageGray[top:bottom, left:right]
        imageCropFix = _cropFix(imageCrop)

        if imageCropFix is not None:
            if isDebug:
                cv2.rectangle(imageGray, (left, top), (right, bottom), (0, 0, 0), 1)

                preprocessorHelper.write(f"{PATH_ROOT}{PATH_FILE_OUTPUT}tesseract/{fileName}", "_box", imageGray)

            imageResult[top:bottom, left:right] = imageCropFix
    
    preprocessorHelper.write(f"{PATH_ROOT}{PATH_FILE_OUTPUT}tesseract/{fileName}", "_result", imageResult)

    resultLanguage = ""
    resultPsm = 6

    if language == "en":
        resultLanguage = "eng"
    elif language == "jp":
        resultLanguage = "jpn"
    elif language == "jp_vert":
        resultLanguage = "jpn_vert"
        resultPsm = 5

    os.environ["TESSDATA_PREFIX"] = f"{PATH_ROOT}src/library/tesseract/language/"
    
    subprocess.run([
        f"{PATH_ROOT}src/library/tesseract/executable",
        f"{PATH_ROOT}{PATH_FILE_OUTPUT}tesseract/{fileNameSplit}_result{fileExtensionSplit}",
        f"{PATH_ROOT}{PATH_FILE_OUTPUT}tesseract/{fileNameSplit}",
        f"-l", resultLanguage,
        "--oem", "1",
        "--psm", str(resultPsm),
        "-c", "preserve_interword_spaces=1",
        "-c", "page_separator=''",
        "-c", "tessedit_char_blacklist=''",
        "-c", "tessedit_create_txt=1",
        "-c", "tessedit_create_hocr=1",
        "-c", "tessedit_create_alto=1",
        "-c", "tessedit_create_page_xml=1",
        "-c", "tessedit_create_lstmbox=1",
        "-c", "tessedit_create_tsv=1",
        "-c", "tessedit_create_wordstrbox=1",
        "-c", "tessedit_create_pdf=1",
        "-c", "tessedit_create_boxfile=1"
    ], check=True)

def test():
    print(f"Load file: {PATH_ROOT}{PATH_FILE_INPUT}{fileName}\r")

    imageRead = preprocessorHelper.read(f"{PATH_ROOT}{PATH_FILE_INPUT}{fileName}")

    imageGray = preprocessorHelper.gray(imageRead)
    
    imageBox = imageGray.copy()

    imageResult = imageGray.copy()
    imageResult.fill(255)

    fileNameSplit, _ = os.path.splitext(fileName)

    with open(f"{PATH_ROOT}{PATH_FILE_OUTPUT}craft/{fileNameSplit}.txt", "r") as file:
        lineList = file.readlines()

    for line in lineList:
        coordinateList = list(map(int, line.strip().split(",")))
        boxList = numpy.array(coordinateList).reshape((-1, 2))

        left = int(boxList[0][0])
        top = int(boxList[0][1])
        right = int(boxList[2][0])
        bottom = int(boxList[2][1])

        imageCrop = imageGray[top:bottom, left:right]
        #imageCropFix = _cropFix(imageCrop)

        if imageCrop is not None:
            if isDebug:
                cv2.rectangle(imageBox, (left, top), (right, bottom), (0, 0, 0), 1)

                preprocessorHelper.write(f"{PATH_ROOT}{PATH_FILE_OUTPUT}tesseract/{fileName}", "_box", imageBox)

            imageResult[top:bottom, left:right] = imageCrop

            preprocessorHelper.write(f"{PATH_ROOT}{PATH_FILE_OUTPUT}tesseract/{fileNameSplit}.png", "_crop", imageCrop)

            _subprocess(fileNameSplit)
    
    preprocessorHelper.write(f"{PATH_ROOT}{PATH_FILE_OUTPUT}tesseract/{fileName}", "_result", imageResult)
