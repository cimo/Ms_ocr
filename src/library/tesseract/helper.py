import sys
import os
import logging
import ast
import numpy
import cv2
from pathlib import Path
from dotenv import load_dotenv

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
        if os.environ.get(varKey).startswith("[") and os.environ.get(varKey).endswith(
            "]"
        ):
            return ast.literal_eval(os.environ.get(varKey))
    return os.environ.get(varKey)

ENV_NAME = checkEnvVariable("ENV_NAME")

def locationFromEnvName():
    result = ENV_NAME.split("_")[-1]

    if result == "local":
        result = "jp"

    return result

dotenvPath = Path(f"../../../env/{ENV_NAME}.env")

if dotenvPath.exists():
    load_dotenv(dotenv_path=dotenvPath)
else:
    logging.exception(f"Environment file {dotenvPath} not found!")

PATH_ROOT = checkEnvVariable("PATH_ROOT")
PATH_FILE_INPUT = checkEnvVariable("MS_O_PATH_FILE_INPUT")
PATH_FILE_OUTPUT = checkEnvVariable("MS_O_PATH_FILE_OUTPUT")

fileName = sys.argv[1]
language = sys.argv[2]
output = sys.argv[3]
isCuda = sys.argv[4]
isDebug = sys.argv[5]

def _writeOutputImage(label, image):
    fileNameSpit, fileExtensionSplit = os.path.splitext(fileName)
    pathJoin = os.path.join(f"{PATH_ROOT}{PATH_FILE_OUTPUT}", f"{fileNameSpit}{label}{fileExtensionSplit}")
    
    imageResult = numpy.clip(image, 0, 255).astype(numpy.uint8)

    cv2.imwrite(pathJoin, imageResult)

def _backgroundCheck(image):
    imageBgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    imageHsv = cv2.cvtColor(imageBgr, cv2.COLOR_BGR2HSV)

    colorMin = numpy.array([0, 0, 0])
    colorMax = numpy.array([179, 255, 146])
    mask = cv2.inRange(imageHsv, colorMin, colorMax)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    dilatation = cv2.dilate(mask, kernel, iterations=5)

    return 255 - cv2.bitwise_and(dilatation, mask)

def _addBorder(image, color, unit):
    return cv2.rectangle(image, (0, image.shape[0]), (image.shape[1], 0), color, unit)

def _binarization(isInverted, image, unitBlur, unitThresholdA, unitThresholdB):
    thresholdBinary = cv2.THRESH_BINARY

    if isInverted:
        thresholdBinary = cv2.THRESH_BINARY_INV

    imageBlur = cv2.GaussianBlur(image, (unitBlur, unitBlur), 0)
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
    kernel = numpy.ones((unit, unit), numpy.uint8)

    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def _cropFix(image):
    backgroundCheckResult = _backgroundCheck(image)
    pixelQuantity = round((backgroundCheckResult > 0).mean(), 1)

    image = _addBorder(image, (255, 255, 255), 2)

    if pixelQuantity == 1.0 or pixelQuantity <= 0.4:
        imageThreshold = _binarization(True, image, 91, 255, 0)

        image = _addBorder(imageThreshold, (255, 255, 255), 3)
    else:
        image = _binarization(False, image, 91, 255, 5)

    image = _noiseRemove(image, 1)

    return image

def loadFile():
    imageLoad = cv2.imread(f"{PATH_ROOT}{PATH_FILE_INPUT}{fileName}")

    if len(imageLoad.shape) == 2:
        imageLoad = cv2.cvtColor(imageLoad, cv2.COLOR_GRAY2BGR)

    if imageLoad.shape[2] == 4:
        imageLoad = imageLoad[:, :, :3]
        
    imageGray = cv2.cvtColor(imageLoad, cv2.COLOR_BGR2GRAY)
    imageRectangle = imageGray.copy()

    imageResult = imageGray.copy()
    imageResult.fill(255)

    return imageGray, imageRectangle, imageResult

def readBoxCoordinatesFromFile():
    resultList = []

    fileNameSpit, _ = os.path.splitext(fileName)

    with open(f"{PATH_ROOT}{PATH_FILE_OUTPUT}{fileNameSpit}.txt", "r") as file:
        lineList = file.readlines()

    for line in lineList:
        coordinateList = list(map(int, line.strip().split(",")))
        boxList = numpy.array(coordinateList).reshape((-1, 2))
        
        resultList.append([boxList])
    
    return resultList

def result(coordinateList, imageGray, imageRectangle, imageResult):
    for coordinate in coordinateList:
        top = int(coordinate[0][0][0])
        left = int(coordinate[0][0][1])
        bottom = int(coordinate[0][2][0])
        right = int(coordinate[0][2][1])

        imageCrop = imageGray[left:right, top:bottom]
        imageCropFix = _cropFix(imageCrop)

        if isDebug:
            cv2.rectangle(imageRectangle, (top, left), (bottom, right), (0, 0, 0), 1)

            _writeOutputImage("_box", imageRectangle)

        imageResult[left:right, top:bottom] = imageCropFix
    
    _writeOutputImage("_result", imageResult)