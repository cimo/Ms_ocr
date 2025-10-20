import os
import io
import cv2
import numpy
from PIL import Image

noiseRemoveModeList = {
    "close": cv2.MORPH_CLOSE,
    "open": cv2.MORPH_OPEN,
    "dilate": cv2.MORPH_DILATE,
    "erode": cv2.MORPH_ERODE
}

def _imageArray(image, isRequestedFloat=False):
    if isinstance(image, numpy.ndarray):
        result = image
    else:
        result = numpy.array(image)

    if isRequestedFloat:
        result = result.astype(numpy.float32)
    else:
        if result.dtype != numpy.uint8:
            result = numpy.clip(result, 0, 255).astype(numpy.uint8)

    return result

def open(pathFull):
    result = Image.open(pathFull)
    result.info["icc_profile"] = result.info.get("icc_profile")

    return _imageArray(result)

def resize(image, sizeLimit):
    imageArray = _imageArray(image)

    height, width = imageArray.shape[:2]

    if height >= width:
        ratioHeight = sizeLimit / height
        ratioWidth = ratioHeight 

        targetHeight = sizeLimit
        targetWidth = int(width * ratioHeight)
    else:
        ratioWidth = sizeLimit / width
        ratioHeight = ratioWidth
        
        targetWidth = sizeLimit
        targetHeight = int(height * ratioWidth)

    if ratioHeight < 1 and ratioWidth < 1:  
        interpolation = cv2.INTER_AREA
    else:  
        interpolation = cv2.INTER_CUBIC

    imageResult = cv2.resize(imageArray, (targetWidth, targetHeight), interpolation=interpolation)

    if imageResult.ndim == 2:
        channel = 1
    else:
        channel = imageResult.shape[2]

    return targetWidth, targetHeight, ratioWidth, ratioHeight, imageResult, channel

def resizeLineHeight(image, heightTarget=30):
    imageGray = gray(image)

    _, imageThreshold = cv2.threshold(imageGray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contourList, _ = cv2.findContours(imageThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    heightList = []

    for contour in contourList:
        _, _, _, height = cv2.boundingRect(contour)
        
        if 8 <= height <= 100:
            heightList.append(height)

    if heightList:
        heightCurrent = int(numpy.median(heightList))
    else:
        heightCurrent = 10

    ratio = heightTarget / heightCurrent

    targetWidth = int(image.shape[1] * ratio)
    targetHeight = int(image.shape[0] * ratio)

    imageResult = cv2.resize(image, (targetWidth, targetHeight), interpolation=cv2.INTER_CUBIC)

    if imageResult.ndim == 2:
        imageResult = numpy.expand_dims(imageResult, axis=-1)

    if imageResult.shape[2] == 1:
        imageResult = numpy.repeat(imageResult, 3, axis=-1)

    _, _, channel = imageResult.shape

    return targetWidth, targetHeight, ratio, imageResult, channel

def gray(image):
    imageArray = _imageArray(image)

    if imageArray.ndim == 3 and imageArray.shape[2] == 3:
        imageArray = cv2.cvtColor(imageArray, cv2.COLOR_RGB2GRAY)
    elif imageArray.ndim == 3 and imageArray.shape[2] == 4:
        imageArray = cv2.cvtColor(imageArray, cv2.COLOR_RGBA2GRAY)
    
    return imageArray

def color(image):
    imageArray = _imageArray(image)

    if imageArray.ndim == 2:
        imageArray = cv2.cvtColor(imageArray, cv2.COLOR_GRAY2BGR)
    if imageArray.ndim == 3 and imageArray.shape[2] == 1:
        imageArray = cv2.cvtColor(imageArray.squeeze(-1), cv2.COLOR_GRAY2BGR)
    if imageArray.ndim == 3 and imageArray.shape[2] == 4:
        imageArray = cv2.cvtColor(imageArray, cv2.COLOR_RGBA2BGR)
    
    return imageArray

def medianBlur(image, value=5):
    imageArray = _imageArray(image)

    return cv2.medianBlur(imageArray, value)

def binarization(image, blur=7, threshold=11, block=2, isInverted=False):
    imageArray = _imageArray(image, True)

    imageBlur = medianBlur(imageArray)

    if isInverted:
        thresholdType = cv2.THRESH_BINARY_INV
    else:
        thresholdType = cv2.THRESH_BINARY
    
    return cv2.adaptiveThreshold(imageBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType, threshold, block)

def contour(image):
    imageArray = _imageArray(image)

    return cv2.findContours(imageArray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def mask(image):
    imageArray = _imageArray(image)

    return numpy.zeros_like(imageArray)

def maskApply(image, imageMask):
    imageArray = _imageArray(image)

    return cv2.bitwise_and(imageArray, imageArray, mask=imageMask)

def gamma(image, gamma=2.2):
    imageArray = _imageArray(image)

    gammaBA = 1.0 / gamma
    
    table = numpy.array([(a / 255.0) ** gammaBA * 255 for a in range(256)]).astype("uint8")
    
    return cv2.LUT(imageArray.astype(numpy.uint8), table)

def noiseRemove(image, modeValue="close", unit=1):
    imageArray = _imageArray(image)
    
    mode = modeValue

    if mode not in noiseRemoveModeList and mode != "auto":
        mode = "auto"
    
    if mode == "auto":
        totalPixel = imageArray.size

        whitePixel = numpy.sum(imageArray > 127)
        blackPixel = totalPixel - whitePixel

        if whitePixel > blackPixel:
            mode = noiseRemoveModeList["close"]
        else:
            mode = noiseRemoveModeList["open"]

        _, label = cv2.connectedComponents((imageArray > 127).astype(numpy.uint8))

        unique, countList = numpy.unique(label, return_counts=True)
        sizeList = countList[unique != 0]

        if len(sizeList) == 0:
            unitAuto = 3
        else:
            unitAuto = int(numpy.sqrt(numpy.mean(sizeList)))

            if unitAuto < 1:
                unitAuto = 1

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (unitAuto, unitAuto))
    else:
        mode = noiseRemoveModeList[mode]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (unit, unit))
    
    return cv2.morphologyEx(imageArray, mode, kernel)

def addOrRemoveBorder(image, unit=1, color=125):
    imageArray = _imageArray(image)

    if unit > 0:
        return cv2.copyMakeBorder(imageArray, top=unit, bottom=unit, left=unit, right=unit, borderType=cv2.BORDER_CONSTANT, value=color)
    else:
        unitNegative = -unit

        height, width = imageArray.shape[:2]

        return imageArray[unitNegative:height-unitNegative, unitNegative:width-unitNegative]

def heatmap(scoreText):
    scoreTextNormalize = cv2.normalize(scoreText, None, 0, 255, cv2.NORM_MINMAX).astype(numpy.uint8)

    return cv2.applyColorMap(scoreTextNormalize, cv2.COLORMAP_JET)

def write(pathFull, label, image, dpi=(300, 300)):
    fileNameSplit, fileExtensionSplit = os.path.splitext(os.path.basename(pathFull))
    dirName = os.path.dirname(pathFull)
    pathJoin = os.path.join(dirName, f"{fileNameSplit}{label}{fileExtensionSplit}")

    os.makedirs(dirName, exist_ok=True)

    if isinstance(image, numpy.ndarray):
        if numpy.issubdtype(image.dtype, numpy.floating):
            image = numpy.clip(image, 0, 255).astype(numpy.uint8)
        elif image.dtype != numpy.uint8:
            image = numpy.clip(image, 0, 255).astype(numpy.uint8)

        if image.ndim == 2:
            image = Image.fromarray(image, mode="L")
        elif image.ndim == 3:
            if image.shape[2] == 1:
                image = Image.fromarray(image.squeeze(-1), mode="L")
            elif image.shape[2] == 3:
                image = Image.fromarray(image, mode="RGB")
            elif image.shape[2] == 4:
                image = Image.fromarray(image, mode="RGBA")
    
    iccProfile = None
    
    if hasattr(image, "info"):
        iccProfile = image.info.get("icc_profile")

    if fileExtensionSplit.lower() in [".jpg", ".jpeg"]:
        image.save(pathJoin, quality=100, subsampling=0, icc_profile=iccProfile, dpi=dpi)
    else:
        image.save(pathJoin, icc_profile=iccProfile, dpi=dpi)

def writeMemory(image, fileName, dpi=(300, 300)):
    fileNameSplit, _ = os.path.splitext(fileName)

    if isinstance(image, numpy.ndarray):
        if numpy.issubdtype(image.dtype, numpy.floating):
            image = numpy.clip(image, 0, 255).astype(numpy.uint8)
        elif image.dtype != numpy.uint8:
            image = numpy.clip(image, 0, 255).astype(numpy.uint8)

        if image.ndim == 2:
            image = Image.fromarray(image, mode="L")
        elif image.ndim == 3:
            if image.shape[2] == 1:
                image = Image.fromarray(image.squeeze(-1), mode="L")
            elif image.shape[2] == 3:
                image = Image.fromarray(image, mode="RGB")
            elif image.shape[2] == 4:
                image = Image.fromarray(image, mode="RGBA")
    
    buffer = io.BytesIO()

    iccProfile = None
    
    if hasattr(image, "info"):
        iccProfile = image.info.get("icc_profile")

    if fileNameSplit.lower() in [".jpg", ".jpeg"]:
        image.save(buffer, quality=100, subsampling=0, icc_profile=iccProfile, dpi=dpi)
    else:
        image.save(buffer, icc_profile=iccProfile, dpi=dpi)
    
    buffer.seek(0)
    
    return buffer
