import os
import logging
import ast
import json
import numpy
import torch
import torch.backends.cudnn as torchBackendCudnn
from torch.autograd import Variable as torchAutogradVariable
from collections import OrderedDict as collectionOrderDict

# Source
from .detector import Detector
from .refine_net import RefineNet
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
DEVICE = _checkEnvVariable("MS_O_DEVICE")

class CraftDetection:
    def _boxCreation(self, scoreText, _, ratioWidth, ratioHeight):
        scoreText = numpy.where(scoreText > 0.2, scoreText, 0)
        #scoreLink = numpy.where(scoreLink > 0.2, scoreLink, 0)

        for a in range(scoreText.shape[0]):
            rowMean = numpy.mean(scoreText[a, :])

            if rowMean < 0.0025:
                scoreText[a, :] = 0
                
        for b in range(scoreText.shape[1]):
            colMean = numpy.mean(scoreText[:, b])

            if colMean < 0.005:
                scoreText[:, b] = 0

        _, binaryTextMap = imageProcessor.threshold(scoreText, self.textThreshold, 1, 0)
        #_, binaryLinkMap = imageProcessor.threshold(scoreLink, self.textThreshold, 1, 0)
        #scoreCombined = numpy.clip(binaryTextMap + binaryLinkMap, 0, 1)

        imageEroded = imageProcessor.erode(binaryTextMap, 2, 1)

        imageDilate = imageProcessor.dilate(imageEroded, 3, 1)

        if self.isDebug:
            imageProcessor.write(f"{PATH_ROOT}{PATH_FILE_OUTPUT}craft/{self.uniqueId}/{self.fileName}", "_dilate", (imageDilate * 255).astype(numpy.uint8))

        mergeBoxTollerance = 10
        mergeBoxRowTollerance = 8

        contourList, _ = imageProcessor.contour(imageDilate)

        boxList = []

        for contour in contourList:
            x, y, w, h = imageProcessor.bbox(contour)

            if w < 4 or h < 4:
                continue

            if w <= 5 or h <= 5:
                centerX = x + w // 2
                centerY = y + h // 2

                newW = int(w * 1.5)
                newH = int(h * 1.5)

                x = max(0, centerX - newW // 2)
                y = max(0, centerY - newH // 2)
                w = newW
                h = newH

            x = int(x / ratioWidth * 2)
            y = int(y / ratioHeight * 2)
            w = int(w / ratioWidth * 2)
            h = int(h / ratioHeight * 2)

            boxList.append((x, y, w, h))

        boxFilterList = []

        for indexA, (x1, y1, w1, h1) in enumerate(boxList):
            isNested = False

            for indexB, (x2, y2, w2, h2) in enumerate(boxList):
                if indexA != indexB:
                    if x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2:
                        isNested = True

                        break

            if not isNested:
                boxFilterList.append((x1, y1, w1, h1))

        boxMegeList = []

        for boxFilter in boxFilterList:
            isAssigned = False

            _, y, _, _ = boxFilter

            for boxMerge in boxMegeList:
                if abs(y - boxMerge[0][1]) <= mergeBoxTollerance:
                    boxMerge.append(boxFilter)

                    isAssigned = True

                    break

            if not isAssigned:
                boxMegeList.append([boxFilter])

        resultBoxList = []

        for boxMerge in boxMegeList:
            boxMerge.sort(key=lambda b: b[0])

            current = boxMerge[0]

            for next in boxMerge[1:]:
                x1, y1, w1, h1 = current
                x2, y2, w2, h2 = next

                if x2 <= x1 + w1 + mergeBoxRowTollerance:
                    newX = min(x1, x2)
                    newY = min(y1, y2)
                    newW = max(x1 + w1, x2 + w2) - newX
                    newH = max(y1 + h1, y2 + h2) - newY

                    current = (newX, newY, newW, newH)
                else:
                    x, y, w, h = current
                    x = max(0, x - 2)
                    y = max(0, y - 2)
                    w = w + 2 * 2
                    h = h + 2 * 2

                    resultBoxList.append((x, y, w, h))
                    
                    current = next

            x, y, w, h = current
            x = max(0, x - 2)
            y = max(0, y - 2)
            w = w + 2 * 2
            h = h + 2 * 2

            resultBoxList.append((x, y, w, h))

        return resultBoxList

    def _result(self, scoreText, scoreLink, imageOpen, ratioWidth, ratioHeight):
        resultMergeList = []

        boxList = self._boxCreation(scoreText, scoreLink, ratioWidth, ratioHeight)
        
        for (x, y, w, h) in boxList:
            imageProcessor.rectangle(imageOpen, x, y, w, h, (0, 0, 255), 1)

            resultMergeList.append({
                "bbox_list": [x, y, w, h]
            })

        if self.isDebug:
            imageProcessor.write(f"{PATH_ROOT}{PATH_FILE_OUTPUT}craft/{self.uniqueId}/{self.fileName}", "_result", imageOpen)

            with open(f"{PATH_ROOT}{PATH_FILE_OUTPUT}craft/{self.uniqueId}/{self.fileNameSplit}_result.json", "w", encoding="utf-8") as file:
                json.dump(resultMergeList, file, ensure_ascii=False, indent=2)
        
        return resultMergeList

    def _normalize(self, image, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
        imageResult = image.copy().astype(numpy.float32)
        imageResult -= numpy.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=numpy.float32)
        imageResult /= numpy.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=numpy.float32)
        
        return imageResult

    def _denormalize(self, image, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
        imageResult = image.copy()
        imageResult = (imageResult * variance + mean) * 255.0
        imageResult = numpy.clip(imageResult, 0, 255).astype(numpy.uint8)

        return imageResult

    def _inference(self, imageResizeCnn, detector, refineNet):
        imageNormalize  = self._normalize(imageResizeCnn)
        imageTensor = torch.from_numpy(imageNormalize).permute(2, 0, 1)
        modelInput = torchAutogradVariable(imageTensor.unsqueeze(0))
        
        if DEVICE == "gpu":
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

        return scoreText, scoreLink

    def _resizeCnn(self, targetWidth, targetHeight, channel, imageResize):
        target32Width = targetWidth
        target32Height = targetHeight
        
        if targetHeight % 32 != 0:
            target32Height = targetHeight + (32 - targetHeight % 32)
        
        if targetWidth % 32 != 0:
            target32Width = targetWidth + (32 - targetWidth % 32)
        
        imageResult = numpy.zeros((target32Height, target32Width, channel), dtype=numpy.float32)
        imageResult[0:targetHeight, 0:targetWidth, :] = imageResize

        return imageResult

    def _preprocess(self):
        imageOpen, _, _ = imageProcessor.open(f"{PATH_ROOT}{PATH_FILE_INPUT}{self.fileName}")

        targetWidth, targetHeight, ratioWidth, ratioHeight, imageResize, channel = imageProcessor.resize(imageOpen, 1280)

        imageGray = imageProcessor.gray(imageResize)

        imageBinarize = imageProcessor.binarization(imageGray)

        imageNoiseRemove = imageProcessor.noiseRemove(imageBinarize)

        imageColor = imageProcessor.color(imageNoiseRemove)

        imageResizeCnn = self._resizeCnn(targetWidth, targetHeight, channel, imageColor)

        if self.isDebug:
            imageProcessor.write(f"{PATH_ROOT}{PATH_FILE_OUTPUT}craft/{self.uniqueId}/{self.fileName}", "_preprocess", imageColor)

        return imageOpen, ratioWidth, ratioHeight, imageResizeCnn

    def _removeDataParallel(self, stateDict):
        if list(stateDict.keys())[0].startswith("module"):
            indexStart = 1
        else:
            indexStart = 0

        stateDictNew = collectionOrderDict()

        for key, value in stateDict.items():
            name = ".".join(key.split(".")[indexStart:])

            stateDictNew[name] = value

        return stateDictNew

    def _refineNetEval(self, refineNet):
        if self.isRefine:
            if DEVICE == "gpu":
                refineNet.load_state_dict(self._removeDataParallel(torch.load(self.pathWeightRefine)))

                refineNet = torch.nn.DataParallel(refineNet.cuda())

                torchBackendCudnn.benchmark = False
            else:
                refineNet.load_state_dict(self._removeDataParallel(torch.load(self.pathWeightRefine, map_location="cpu")))

            refineNet.eval()

            return refineNet
        else:
            return None

    def _detectorEval(self, detector):
        if DEVICE == "gpu":
            detector.load_state_dict(self._removeDataParallel(torch.load(self.pathWeightMain)))

            detector = torch.nn.DataParallel(detector.cuda())
            
            torchBackendCudnn.benchmark = False
        else:
            detector.load_state_dict(self._removeDataParallel(torch.load(self.pathWeightMain, map_location="cpu")))

        detector.eval()

        return detector

    def __init__(self, fileNameValue, isDebugValue, uniqueIdValue):
        self.fileName = fileNameValue
        self.isDebug = isDebugValue
        self.uniqueId = uniqueIdValue

        self.pathWeightMain = f"{PATH_ROOT}src/library/craft_detection/mlt_25k.pth"
        self.pathWeightRefine = f"{PATH_ROOT}src/library/craft_detection/refiner_CTW1500.pth"
        self.textThreshold = 0.1
        #self.linkThreshold = 0.1
        self.isRefine = False
        self.resultMainList = None

        self.fileNameSplit = ".".join(self.fileName.split(".")[:-1])

        detector = Detector()
        detector = self._detectorEval(detector)

        refineNet = None

        if self.isRefine:
            refineNet = RefineNet()
            refineNet = self._refineNetEval(refineNet)

        imageOpen, ratioWidth, ratioHeight, imageResizeCnn = self._preprocess()

        scoreText, scoreLink = self._inference(imageResizeCnn, detector, refineNet)

        self.resultMainList = self._result(scoreText, scoreLink, imageOpen, ratioWidth, ratioHeight)
    
    def getResultMainList(self):
        return self.resultMainList
