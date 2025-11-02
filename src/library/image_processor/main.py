import os
import io
import numpy
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

noiseRemoveModeList = {
    "close": cv2.MORPH_CLOSE,
    "open": cv2.MORPH_OPEN,
    "dilate": cv2.MORPH_DILATE,
    "erode": cv2.MORPH_ERODE
}

def _fileArray(file, isFloat=False):
    if isinstance(file, numpy.ndarray):
        result = file
    else:
        result = numpy.array(file)

    if isFloat:
        result = result.astype(numpy.float32)
    else:
        if result.dtype != numpy.uint8:
            result = numpy.clip(result, 0, 255).astype(numpy.uint8)

    return result

def pilImage(file):
    inputHeight, inputWidth = file.shape[:2]
    imageNew = Image.new("RGB", (inputWidth, inputHeight), (255, 255, 255))
    imageDraw = ImageDraw.Draw(imageNew)

    return imageNew, imageDraw

def pilFont(text, width, height, fontName):
    fontScaleMax = 1 if len(text) <= 2 else 0.8
    fontSize = min(height, int(height * fontScaleMax))

    while fontSize > 1:
        pilFont = ImageFont.truetype(fontName, fontSize)
        bboxText = pilFont.getbbox(text)

        textWidth = bboxText[2] - bboxText[0]
        textHeight = bboxText[3] - bboxText[1]

        if textWidth <= width and textHeight <= height:
            break

        fontSize -= 1
    
    return pilFont

def open(pathFull):
    result = Image.open(pathFull).convert("RGB")
    pilIccProfile = result.info.get("icc_profile")
    pilExif = result.info.get("exif")

    return _fileArray(result), pilIccProfile, pilExif

def resizeMultiple(file, multiple=32):
    file = Image.fromarray(file)
    
    width, height = file.size

    widthResult = width % multiple
    widthDown = width - widthResult
    widthUp = widthDown + multiple
    widthSnap = widthUp if widthResult >= (multiple / 2) else widthDown
    
    if widthSnap < multiple:
        widthSnap = multiple

    ratioOriginal = width / float(height)
    heightResult = int(round(widthSnap / ratioOriginal))
    heightDown = max(multiple, heightResult - (heightResult % multiple))
    heightUp = heightDown + multiple
    heightStart = heightUp if (heightResult - heightDown) >= (heightUp - heightResult) else heightDown
    heightEnd = heightStart

    errorMin = abs((widthSnap / heightStart) - ratioOriginal) / max(1e-9, ratioOriginal)

    for a in range(1, 11):
        upCandidate = heightStart + a * multiple
        upError = abs((widthSnap / upCandidate) - ratioOriginal) / max(1e-9, ratioOriginal)

        if upError < errorMin:
            errorMin = upError
            heightEnd = upCandidate

        downCandidate = max(multiple, heightStart - a * multiple)
        downError = abs((widthSnap / downCandidate) - ratioOriginal) / max(1e-9, ratioOriginal)
        
        if downError < errorMin:
            errorMin = downError
            heightEnd = downCandidate
    
    heightSnap = heightEnd

    if widthSnap % 2 == 1:
        widthSnap += 1

    if heightSnap % 2 == 1:
        heightSnap += 1

    scaleX = widthSnap / float(width)
    scaleY = heightSnap / float(height)
    scaleMax = max(scaleX, scaleY)

    if scaleX < 1.0 or scaleY < 1.0:
        resample = Image.LANCZOS
        resampleName = "LANCZOS"
    elif scaleMax <= 2.0:
        resample = Image.BICUBIC
        resampleName = "BICUBIC"
    else:
        resample = Image.LANCZOS
        resampleName = "LANCZOS"
    
    resized = file.resize((widthSnap, heightSnap), resample)
    
    if scaleX < 1.0 or scaleY < 1.0:
        resized = resized.filter(ImageFilter.SHARPEN)

    result = numpy.array(resized)

    return {
        "sizeOriginal": (width, height),
        "sizeNew": (widthSnap, heightSnap),
        "ratioError": errorMin * 100.0,
        "scaleX": scaleX,
        "scaleY": scaleY,
        "scaleMax": scaleMax,
        "resampleName": resampleName,
        "result": result
    }

def resize(file, sizeLimit, side="w"):
    fileArray = _fileArray(file)

    height, width = fileArray.shape[:2]

    if side == "w":
        ratio = sizeLimit / width
        
        targetWidth = sizeLimit
        targetHeight = int(height * ratio)
    elif side == "h":
        ratio = sizeLimit / height

        targetHeight = sizeLimit
        targetWidth = int(width * ratio)

    if ratio < 1.0:
        interpolation = cv2.INTER_LANCZOS4
    else:
        interpolation = cv2.INTER_CUBIC

    imageResult = cv2.resize(fileArray, (targetWidth, targetHeight), interpolation=interpolation)

    if ratio < 1.0:
        kernelSharpening = numpy.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        imageResult = cv2.filter2D(imageResult, -1, kernelSharpening)

    channel = 1 if imageResult.ndim == 2 else imageResult.shape[2]

    return targetWidth, targetHeight, ratio, imageResult, channel

def rgbToGray(file):
    fileArray = _fileArray(file)

    if fileArray.ndim == 3 and fileArray.shape[2] == 3:
        fileArray = cv2.cvtColor(fileArray, cv2.COLOR_RGB2GRAY)
    elif fileArray.ndim == 3 and fileArray.shape[2] == 4:
        fileArray = cv2.cvtColor(fileArray, cv2.COLOR_RGBA2GRAY)
    
    return fileArray

def grayToRgb(file):
    fileArray = _fileArray(file)

    if fileArray.ndim == 2:
        fileArray = cv2.cvtColor(fileArray, cv2.COLOR_GRAY2RGB)
    if fileArray.ndim == 3 and fileArray.shape[2] == 1:
        fileArray = cv2.cvtColor(fileArray.squeeze(-1), cv2.COLOR_GRAY2RGB)

    return fileArray

def medianBlur(file, value=5):
    fileArray = _fileArray(file)

    return cv2.medianBlur(fileArray, value)

def binarization(file, threshold=11, block=2, isInverted=False):
    fileArray = _fileArray(file, True)

    imageBlur = medianBlur(fileArray)

    if isInverted:
        thresholdType = cv2.THRESH_BINARY_INV
    else:
        thresholdType = cv2.THRESH_BINARY
    
    return cv2.adaptiveThreshold(imageBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType, threshold, block)

def threshold(file, value, valueMax, type):
    fileArray = _fileArray(file, True)

    return cv2.threshold(fileArray, value, valueMax, type)

def erode(file, kernelWidth, kernelHeight):
    fileArray = _fileArray(file)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelWidth, kernelHeight))
    
    return cv2.erode(fileArray, kernel)

def dilate(file, kernelWidth, kernelHeight):
    fileArray = _fileArray(file)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelWidth, kernelHeight))
    
    return cv2.dilate(fileArray, kernel)

def contour(file):
    fileArray = _fileArray(file)

    return cv2.findContours(fileArray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def bbox(contour):
    return cv2.boundingRect(contour)

def rectangle(file, x, y, w, h, color, border):
    return cv2.rectangle(file, (x, y), (x + w, y + h), color, border)

def mask(file):
    fileArray = _fileArray(file)

    return numpy.zeros_like(fileArray)

def maskApply(file, imageMask):
    fileArray = _fileArray(file)

    return cv2.bitwise_and(fileArray, fileArray, mask=imageMask)

def gamma(file, value=2.2):
    fileArray = _fileArray(file)

    table = numpy.array([(a / 255.0) ** (1.0 / value) * 255 for a in range(256)])
    
    return cv2.LUT(fileArray, table)

def noiseRemove(file, modeValue="close", unit=1):
    fileArray = _fileArray(file)
    
    mode = modeValue

    if mode not in noiseRemoveModeList and mode != "auto":
        mode = "auto"
    
    if mode == "auto":
        totalPixel = fileArray.size

        whitePixel = numpy.sum(fileArray > 127)
        blackPixel = totalPixel - whitePixel

        if whitePixel > blackPixel:
            mode = noiseRemoveModeList["close"]
        else:
            mode = noiseRemoveModeList["open"]

        _, label = cv2.connectedComponents((fileArray > 127))

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
    
    return cv2.morphologyEx(fileArray, mode, kernel)

def addOrRemoveBorder(file, unit=1, color=125):
    fileArray = _fileArray(file)

    if unit > 0:
        return cv2.copyMakeBorder(fileArray, top=unit, bottom=unit, left=unit, right=unit, borderType=cv2.BORDER_CONSTANT, value=color)
    else:
        unitNegative = -unit

        height, width = fileArray.shape[:2]

        return fileArray[unitNegative:height-unitNegative, unitNegative:width-unitNegative]

def heatmap(scoreText):
    scoreTextNormalize = cv2.normalize(scoreText, None, 0, 255, cv2.NORM_MINMAX)

    return cv2.applyColorMap(scoreTextNormalize, cv2.COLORMAP_JET)

def write(pathFull, label, file, pilIccProfile=None, pilExif=None, dpi=(300, 300)):
    fileNameSplit, fileExtensionSplit = os.path.splitext(os.path.basename(pathFull))
    dirName = os.path.dirname(pathFull)
    pathJoin = os.path.join(dirName, f"{fileNameSplit}{label}{fileExtensionSplit}")

    os.makedirs(dirName, exist_ok=True)

    if isinstance(file, numpy.ndarray):
        if numpy.issubdtype(file.dtype, numpy.floating):
            file = numpy.clip(file, 0, 255).astype(numpy.uint8)
        elif file.dtype != numpy.uint8:
            file = numpy.clip(file, 0, 255).astype(numpy.uint8)

        if file.ndim == 2:
            file = Image.fromarray(file, mode="L")
        elif file.ndim == 3:
            if file.shape[2] == 1:
                file = Image.fromarray(file.squeeze(-1), mode="L")
            elif file.shape[2] == 3:
                file = Image.fromarray(file, mode="RGB")
            elif file.shape[2] == 4:
                file = Image.fromarray(file, mode="RGBA")
    
    parameterList = {"icc_profile": pilIccProfile, "dpi": dpi}

    if pilExif is not None:
        parameterList["exif"] = pilExif

    if fileExtensionSplit.lower() in [".jpg", ".jpeg"]:
        parameterList["quality"] = 100
        parameterList["subsampling"] = 0

    file.save(pathJoin, **parameterList)

def writeMemory(file, fileName, pilIccProfile=None, pilExif=None, dpi=(300, 300)):
    _, fileExtensionSplit = os.path.splitext(fileName)

    if isinstance(file, numpy.ndarray):
        if numpy.issubdtype(file.dtype, numpy.floating):
            file = numpy.clip(file, 0, 255).astype(numpy.uint8)
        elif file.dtype != numpy.uint8:
            file = numpy.clip(file, 0, 255).astype(numpy.uint8)

        if file.ndim == 2:
            file = Image.fromarray(file, mode="L")
        elif file.ndim == 3:
            if file.shape[2] == 1:
                file = Image.fromarray(file.squeeze(-1), mode="L")
            elif file.shape[2] == 3:
                file = Image.fromarray(file, mode="RGB")
            elif file.shape[2] == 4:
                file = Image.fromarray(file, mode="RGBA")
    
    buffer = io.BytesIO()

    parameterList = {"icc_profile": pilIccProfile, "dpi": dpi}

    if pilExif is not None:
        parameterList["exif"] = pilExif

    if fileExtensionSplit.lower() in [".jpg", ".jpeg"]:
        parameterList["quality"] = 100
        parameterList["subsampling"] = 0

    file.save(buffer, **parameterList)
    
    buffer.seek(0)
    
    return buffer
