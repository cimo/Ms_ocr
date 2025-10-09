import sys
import os
import time
import cv2
import numpy
import math
import torch
import torch.backends.cudnn as torchBackendCudnn
from torch.autograd import Variable as torchAutogradVariable
from collections import OrderedDict as collectionOrderDict

# Source
from preprocessor import main as preprocessor

pathRoot = sys.argv[1]
pathInput = sys.argv[2]
pathOutput = sys.argv[3]
fileName = sys.argv[4]
isCuda = sys.argv[5].lower() == "true"
isDebug = sys.argv[6].lower() == "true"

pathWeightMain = os.path.join(os.path.dirname(__file__), "mlt_25k.pth")
pathWeightRefine = os.path.join(os.path.dirname(__file__), "refiner_CTW1500.pth")
lowText = 0.4
thresholdText = 0.7
thresholdLink = 0.1
isRefine = True

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

def _resizeCnn(targetWidth, targetHeight, channel, imageResize):
    target32Width = targetWidth
    target32Height = targetHeight
    
    if targetHeight % 32 != 0:
        target32Height = targetHeight + (32 - targetHeight % 32)
    
    if targetWidth % 32 != 0:
        target32Width = targetWidth + (32 - targetWidth % 32)
    
    imageResult = numpy.zeros((target32Height, target32Width, channel), dtype=numpy.float32)
    imageResult[0:targetHeight, 0:targetWidth, :] = imageResize

    return imageResult

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

def _boxCreation(scoreTextValue, scoreLinkValue, ratio):
    if isDebug:
        imageHeatmapScoreText = preprocessor.heatmap(scoreTextValue)
        imageHeatmapScoreLink = preprocessor.heatmap(scoreLinkValue)

        preprocessor.write(f"{pathRoot}{pathOutput}{fileName}", "_heatmap_text", imageHeatmapScoreText)
        preprocessor.write(f"{pathRoot}{pathOutput}{fileName}", "_heatmap_link", imageHeatmapScoreLink)

    imageHeight, imageWidth = scoreTextValue.shape

    width = int(imageWidth * ratio)
    height = int(imageHeight * ratio)

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

        box /= ratio
        box *= ((1 / ratio) * 2, (1 / ratio) * 2)

        boxList.append(box)

    return numpy.array(boxList)

def checkCuda():
    print(f"On your machine CUDA are: {'available' if torch.cuda.is_available() else 'NOT available'}.")

def detectorEval(detector):
    print(f"Loading weight: {pathWeightMain}")

    if isCuda:
        detector.load_state_dict(_removeDataParallel(torch.load(pathWeightMain)))

        detector = torch.nn.DataParallel(detector.cuda())
        
        torchBackendCudnn.benchmark = False
    else:
        detector.load_state_dict(_removeDataParallel(torch.load(pathWeightMain, map_location="cpu")))

    detector.eval()

    return detector

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
    print(f"Load file: {pathRoot}{pathInput}{fileName}\r")

    imageRead = preprocessor.open(f"{pathRoot}{pathInput}{fileName}")

    targetWidth, targetHeight, ratio, imageResize, channel = preprocessor.resize(imageRead, 2048)

    imageGray = preprocessor.gray(imageResize)

    imageBinarize = preprocessor.binarization(imageGray)

    imageNoiseRemove = preprocessor.noiseRemove(imageBinarize)

    imageColor = preprocessor.color(imageNoiseRemove)

    imageResizeCnn = _resizeCnn(targetWidth, targetHeight, channel, imageColor)

    if isDebug:
        preprocessor.write(f"{pathRoot}{pathOutput}{fileName}", "_preprocess", imageColor)

    return ratio, imageResizeCnn, imageRead

def inference(imageResizeCnn, detector, refineNet):
    timeStart = time.time()

    imageNormalize  = _normalize(imageResizeCnn)
    imageTensor = torch.from_numpy(imageNormalize).permute(2, 0, 1)
    modelInput = torchAutogradVariable(imageTensor.unsqueeze(0))
    
    if isCuda:
        modelInput = modelInput.cuda()

    with torch.no_grad():
        scoreMap, feature = detector(modelInput)

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

def result(scoreText, scoreLink, ratio, imageRead):
    boxList = _boxCreation(scoreText, scoreLink, ratio)

    fileNameSplit, _ = os.path.splitext(fileName)

    with open(f"{pathRoot}{pathOutput}{fileNameSplit}.txt", "w") as file:
        for _, box in enumerate(boxList):
            shapeList = numpy.array(box).astype(numpy.int32).reshape((-1))

            file.write(",".join([str(shape) for shape in shapeList]) + "\r\n")

            cv2.polylines(imageRead, [shapeList.reshape(-1, 2).reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=1)

    if isDebug:
        preprocessor.write(f"{pathRoot}{pathOutput}{fileName}", "_result", imageRead)
