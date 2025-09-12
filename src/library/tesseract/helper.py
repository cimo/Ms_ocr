import sys
import os
import subprocess
import logging
import ast
import numpy
import cv2

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

def _loadImage():
    print(f"Load file: {PATH_ROOT}{PATH_FILE_INPUT}{fileName}\r")

    os.makedirs(f"{PATH_ROOT}{PATH_FILE_OUTPUT}tesseract/", exist_ok=True)

    imageLoad = cv2.imread(f"{PATH_ROOT}{PATH_FILE_INPUT}{fileName}")

    if len(imageLoad.shape) == 2:
        imageLoad = cv2.cvtColor(imageLoad, cv2.COLOR_GRAY2BGR)

    if imageLoad.shape[2] == 4:
        imageLoad = imageLoad[:, :, :3]
    
    return imageLoad

def _imageResize(image):
    height, width, _ = image.shape

    sideMin = min(height, width)
    sideMax = max(height, width)

    if sideMin < sizeMax:
        ratio = sizeMax / sideMin
    else:
        ratio = ratioMultiplier * sideMax / sideMax

    targetWidth = int(width * ratio)
    targetHeight = int(height * ratio)

    imageResult = cv2.resize(image, (targetWidth, targetHeight), interpolation=cv2.INTER_LINEAR)

    return imageResult, ratio

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

def _writeOutputImage(label, image):
    fileNameSplit, fileExtensionSplit = os.path.splitext(fileName)
    pathJoin = os.path.join(f"{PATH_ROOT}{PATH_FILE_OUTPUT}tesseract/", f"{fileNameSplit}{label}{fileExtensionSplit}")
    
    imageResult = numpy.clip(image, 0, 255).astype(numpy.uint8)

    cv2.imwrite(pathJoin, imageResult)

def executeCraft():
    subprocess.run([
        "python3",
        f"{PATH_ROOT}src/library/craft/main.py",
        PATH_ROOT,
        PATH_FILE_INPUT,
        PATH_FILE_OUTPUT,
        fileName,
        isCuda,
        isDebug
    ], check=True)

def preprocess():
    imageLoad = _loadImage()

    imageResize, ratio = _imageResize(imageLoad)
        
    imageGray = cv2.cvtColor(imageResize, cv2.COLOR_BGR2GRAY)
    
    imageRectangle = imageGray.copy()

    imageResult = imageGray.copy()
    imageResult.fill(255)

    return imageGray, imageRectangle, imageResult, ratio

def readBoxCoordinateFromFile():
    resultList = []

    fileNameSplit, _ = os.path.splitext(fileName)

    with open(f"{PATH_ROOT}{PATH_FILE_OUTPUT}craft/{fileNameSplit}.txt", "r") as file:
        lineList = file.readlines()

    for line in lineList:
        coordinateList = list(map(int, line.strip().split(",")))
        boxList = numpy.array(coordinateList).reshape((-1, 2))
        
        resultList.append([boxList])
    
    return resultList

def result(coordinateList, ratio, imageGray, imageRectangle, imageResult):
    for coordinate in coordinateList:
        top = int(coordinate[0][0][0] * ratio)
        left = int(coordinate[0][0][1] * ratio)
        bottom = int(coordinate[0][2][0] * ratio)
        right = int(coordinate[0][2][1] * ratio)

        imageCrop = imageGray[left:right, top:bottom]
        imageCropFix = _cropFix(imageCrop)

        if imageCropFix is not None:
            if isDebug:
                cv2.rectangle(imageRectangle, (top, left), (bottom, right), (0, 0, 0), 1)

                _writeOutputImage("_box", imageRectangle)

            imageResult[left:right, top:bottom] = imageCropFix
    
    _writeOutputImage("_result", imageResult)

def execute():
    resultLanguage = ""
    resultPsm = 6

    if language == "en":
        resultLanguage = "eng"
    elif language == "jp":
        resultLanguage = "jpn"
    elif language == "jp_vert":
        resultLanguage = "jpn_vert"
        resultPsm = 5

    fileNameSplit, fileExtensionSplit = os.path.splitext(fileName)

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
    # Coordinate
    resultList = []

    fileNameSplit, _ = os.path.splitext(fileName)

    with open(f"{PATH_ROOT}{PATH_FILE_OUTPUT}craft/{fileNameSplit}.txt", "r") as file:
        lineList = file.readlines()

    for line in lineList:
        coordinateList = list(map(int, line.strip().split(",")))
        boxList = numpy.array(coordinateList).reshape((-1, 2))
        
        resultList.append([boxList])

    fileNameSplit, fileExtensionSplit = os.path.splitext(fileName)

    # Load
    print(f"Load file: {PATH_ROOT}{PATH_FILE_INPUT}{fileName}\r")

    os.makedirs(f"{PATH_ROOT}{PATH_FILE_OUTPUT}tesseract/", exist_ok=True)

    imageLoad = cv2.imread(f"{PATH_ROOT}{PATH_FILE_INPUT}{fileName}")

    if len(imageLoad.shape) == 2:
        imageLoad = cv2.cvtColor(imageLoad, cv2.COLOR_GRAY2BGR)

    if imageLoad.shape[2] == 4:
        imageLoad = imageLoad[:, :, :3]

    # Resize
    height, width, _ = imageLoad.shape

    sideMin = min(height, width)
    sideMax = max(height, width)

    if sideMin < sizeMax:
        ratio = sizeMax / sideMin
    else:
        ratio = ratioMultiplier * sideMax / sideMax

    print(ratio)

    targetWidth = int(width * ratio)
    targetHeight = int(height * ratio)

    imageResult = cv2.resize(imageLoad, (targetWidth, targetHeight), interpolation=cv2.INTER_LINEAR)

    # Crop
    for a in range(len(resultList)):
        box  = resultList[a]

        top = int(box[0][0][0] * ratio)
        left = int(box[0][0][1] * ratio)
        bottom = int(box[0][2][0] * ratio)
        right = int(box[0][2][1] * ratio)

        imageCrop = imageResult[left:right, top:bottom]

        cropPath = f"{PATH_ROOT}{PATH_FILE_OUTPUT}tesseract/{fileNameSplit}_box_{a}{fileExtensionSplit}"
        cv2.imwrite(cropPath, imageCrop)

        os.environ["TESSDATA_PREFIX"] = f"{PATH_ROOT}src/library/tesseract/language/"

        subprocess.run([
            f"{PATH_ROOT}src/library/tesseract/executable",
            cropPath,
            f"{PATH_ROOT}{PATH_FILE_OUTPUT}tesseract/{fileNameSplit}_box_{a}",
            "-l", "jpn",
            "--oem", "1",
            "--psm", "6",
            "-c", "preserve_interword_spaces=1",
            "-c", "page_separator=''",
            "-c", "tessedit_char_blacklist=''",
            "-c", "tessedit_create_txt=1"
        ], check=True)