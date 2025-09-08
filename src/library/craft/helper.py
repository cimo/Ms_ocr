import sys
import os
import time
import cv2
import numpy
import math
import torch
import torch.backends.cudnn as torchBackendCudnn
from torch.autograd import Variable as torchAutograd
from collections import OrderedDict as collectionOrderDict

pathRoot = sys.argv[1]
pathInput = sys.argv[2]
pathOutput = sys.argv[3]
fileName = sys.argv[4]
pathWeightMain = os.path.join(os.path.dirname(__file__), "craft_mlt_25k.pth")
pathWeightRefine = os.path.join(os.path.dirname(__file__), "craft_refiner_CTW1500.pth")
sizeMax = 2048
ratioMultiplier = 4.0
lowText = 0.2
thresholdText = 0.2
thresholdLink = 0.6
isCuda=sys.argv[5].lower() == "true"
isRefine=True
isDebug=sys.argv[6].lower() == "true"

def _writeOutputImage(label, image):
    fileNameSpit, fileExtensionSplit = os.path.splitext(fileName)
    path = os.path.join(f"{pathRoot}{pathOutput}", f"{fileNameSpit}{label}{fileExtensionSplit}")
    
    if not isinstance(image, tuple):
        imageResult = numpy.clip(image, 0, 255).astype(numpy.uint8)
    else:
        scoreTextValue, scoreLinkValue = image
        imageHstack = numpy.hstack((scoreTextValue, scoreLinkValue))
        imageWrite = (numpy.clip(imageHstack, 0, 1) * 255).astype(numpy.uint8)
        imageResult = cv2.applyColorMap(imageWrite, cv2.COLORMAP_JET)

    cv2.imwrite(path, imageResult)

def _imageResize(image):
    height, width, channel = image.shape

    size = ratioMultiplier * max(height, width)

    if size > sizeMax:
        size = sizeMax

    ratio = size / max(height, width)
    ratioWidth = 1 / ratio
    ratioHeight = 1 / ratio

    targetWidth = int(width * ratio)
    targetHeight = int(height * ratio)

    imageResize = cv2.resize(image, (targetWidth, targetHeight), interpolation=cv2.INTER_LINEAR)

    target32Width = targetWidth
    target32Height = targetHeight
    
    if targetHeight % 32 != 0:
        target32Height = targetHeight + (32 - targetHeight % 32)
    
    if targetWidth % 32 != 0:
        target32Width = targetWidth + (32 - targetWidth % 32)
    
    imageResult = numpy.zeros((target32Height, target32Width, channel), dtype=numpy.float32)
    imageResult[0:targetHeight, 0:targetWidth, :] = imageResize

    return imageResult, ratioWidth, ratioHeight

def _normalize(image, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    imageResult = image.copy().astype(numpy.float32)
    imageResult -= numpy.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=numpy.float32)
    imageResult /= numpy.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=numpy.float32)
    
    return imageResult

def _denormalize(image, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    imageResult = image.copy()
    imageResult = (imageResult * variance + mean) * 255.0
    imageResult = numpy.clip(imageResult, 0, 255).astype(numpy.uint8)

    return imageResult

def _boxDetection(scoreTextValue, scoreLinkValue, ratioWidth, ratioHeight):
    if isDebug:
        _writeOutputImage("_heatmap", (scoreTextValue, scoreLinkValue))

    imageHeight, imageWidth = scoreTextValue.shape

    scaleFactor = max(4096 / max(imageHeight, imageWidth), 1.0)
    width = int(imageWidth * scaleFactor)
    height = int(imageHeight * scaleFactor)

    scoreText = cv2.resize(scoreTextValue, (width, height), interpolation=cv2.INTER_LINEAR)
    scoreLink = cv2.resize(scoreLinkValue, (width, height), interpolation=cv2.INTER_LINEAR)

    imageHeight, imageWidth = scoreText.shape

    _, binaryTextMap = cv2.threshold(scoreText, lowText, 1, 0)
    _, binaryLinkMap = cv2.threshold(scoreLink, thresholdLink, 1, 0)

    textScoreCombined = numpy.clip(binaryTextMap + binaryLinkMap, 0, 1)

    labelCount, componentLabelMap, componentStatList, _ = cv2.connectedComponentsWithStats(textScoreCombined.astype(numpy.uint8), connectivity=4)

    boxList = []

    for index in range(1, labelCount):
        size = componentStatList[index, cv2.CC_STAT_AREA]

        if size < 10:
            continue
        
        if numpy.max(scoreText[componentLabelMap == index]) < thresholdText:
            continue

        segmentMap = numpy.zeros(scoreText.shape, dtype=numpy.uint8)
        segmentMap[componentLabelMap == index] = 255
        segmentMap[numpy.logical_and(binaryLinkMap == 1, binaryTextMap == 0)] = 0

        bBoxX, bBoxY = componentStatList[index, cv2.CC_STAT_LEFT], componentStatList[index, cv2.CC_STAT_TOP]
        bBoxWidth, bBoxHeight = componentStatList[index, cv2.CC_STAT_WIDTH], componentStatList[index, cv2.CC_STAT_HEIGHT]
        paddingSize = int(math.sqrt(size * min(bBoxWidth, bBoxHeight) / (bBoxWidth * bBoxHeight)) * 2)

        startX, endX = max(bBoxX - paddingSize, 0), min(bBoxX + bBoxWidth + paddingSize + 1, imageWidth)
        startY, endY = max(bBoxY - paddingSize, 0), min(bBoxY + bBoxHeight + paddingSize + 1, imageHeight)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + paddingSize, 1 + paddingSize))
        segmentMap[startY:endY, startX:endX] = cv2.dilate(segmentMap[startY:endY, startX:endX], kernel)

        contour = numpy.roll(numpy.array(numpy.where(segmentMap != 0)), 1, axis=0).transpose().reshape(-1, 2)

        if contour.shape[0] == 0:
            continue
        
        xMin, xMax = min(contour[:, 0]), max(contour[:, 0])
        yMin, yMax = min(contour[:, 1]), max(contour[:, 1])
        box = numpy.array([[xMin, yMin], [xMax, yMin], [xMax, yMax], [xMin, yMax]], dtype=numpy.float32)

        indexStart = box.sum(axis=1).argmin()
        box = numpy.roll(box, 4 - indexStart, 0)

        box /= scaleFactor

        boxList.append(box)
    
    boxList = numpy.array(boxList)

    for index in range(len(boxList)):
        if boxList[index] is not None:
            boxList[index] *= (ratioWidth * 2, ratioHeight * 2)

    return boxList

def _removeDataParallel(stateDict):
    if list(stateDict.keys())[0].startswith("module"):
        indexStart = 1
    else:
        indexStart = 0

    stateDictNew = collectionOrderDict()

    for key, value in stateDict.items():
        name = ".".join(key.split(".")[indexStart:])

        stateDictNew[name] = value

    return stateDictNew

def craftEval(craft):
    print(f"Loading weight: {pathWeightMain}")

    if isCuda:
        print(isCuda)
        craft.load_state_dict(_removeDataParallel(torch.load(pathWeightMain)))

        craft = torch.nn.DataParallel(craft.cuda())
        
        torchBackendCudnn.benchmark = False
    else:
        craft.load_state_dict(_removeDataParallel(torch.load(pathWeightMain, map_location="cpu")))

    craft.eval()

    return craft

def refineNetEval(refineNet):
    if isRefine:
        print(f"Loading weight refineNet: {pathWeightRefine}")

        if isCuda:
            refineNet.load_state_dict(_removeDataParallel(torch.load(pathWeightRefine)))

            refineNet = torch.nn.DataParallel(refineNet.cuda())

            torchBackendCudnn.benchmark = False
        else:
            refineNet.load_state_dict(_removeDataParallel(torch.load(pathWeightRefine, map_location="cpu")))

        refineNet.eval()

        return refineNet
    else:
        return None

def preprocess():
    os.makedirs(f"{pathRoot}{pathOutput}", exist_ok=True)

    imageLoad = cv2.imread(f"{pathRoot}{pathInput}{fileName}")

    if len(imageLoad.shape) == 2:
        imageLoad = cv2.cvtColor(imageLoad, cv2.COLOR_GRAY2BGR)

    if imageLoad.shape[2] == 4:
        imageLoad = imageLoad[:, :, :3]

    imageGray = cv2.cvtColor(imageLoad, cv2.COLOR_BGR2GRAY)

    denoise = cv2.medianBlur(imageGray, 1)

    threshold = cv2.adaptiveThreshold(denoise, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10)

    invertAB = cv2.bitwise_not(threshold)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

    clean = cv2.morphologyEx(invertAB, cv2.MORPH_OPEN, kernel, iterations=0)

    invertBA = cv2.bitwise_not(clean)

    imageColor = cv2.cvtColor(invertBA, cv2.COLOR_GRAY2BGR)

    imageResize, ratioWidth, ratioHeight = _imageResize(imageColor)

    if isDebug:
        _writeOutputImage("_preprocess", imageResize)

    return imageColor, imageResize, ratioWidth, ratioHeight

def inference(imageValue, craft, refineNet):
    timeStart = time.time()

    image = numpy.array(imageValue)
    imageNormalize  = _normalize(image)
    imageTensor = torch.from_numpy(imageNormalize).permute(2, 0, 1)
    modelInput = torchAutograd(imageTensor.unsqueeze(0))
    
    if isCuda:
        modelInput = modelInput.cuda()

    with torch.no_grad():
        scoreMap, feature = craft(modelInput)

    scoreTextTensor = scoreMap[0, :, :, 0]
    scoreLinkTensor = scoreMap[0, :, :, 1]

    scoreText = scoreTextTensor.cpu().numpy()
    scoreLink = scoreLinkTensor.cpu().numpy()

    if refineNet is not None:
        with torch.no_grad():
            refineMap = refineNet(scoreMap, feature)
        
        scoreLinkTensor = refineMap[0, :, :, 0]
        scoreLink = scoreLinkTensor.cpu().numpy()
    
    print(f"Time inference: {time.time() - timeStart}")

    return scoreText, scoreLink

def result(scoreText, scoreLink, ratioWidth, ratioHeight, image):
    boxList = _boxDetection(scoreText, scoreLink, ratioWidth, ratioHeight)

    fileNameSpit, _ = os.path.splitext(fileName)

    with open(f"{pathRoot}{pathOutput}{fileNameSpit}.txt", "w") as file:
        for _, box in enumerate(boxList):
            shapeList = numpy.array(box).astype(numpy.int32).reshape((-1))

            file.write(",".join([str(shape) for shape in shapeList]) + "\r\n")

            cv2.polylines(image, [shapeList.reshape(-1, 2).reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=1)

    _writeOutputImage("", image)
