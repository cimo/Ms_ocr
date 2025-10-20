import os
import logging
import ast
import numpy
import cv2
import torch
import torch.backends.cudnn as torchBackendCudnn
from torch.autograd import Variable as torchAutogradVariable
from collections import OrderedDict as collectionOrderDict

# Source
from .detector import Detector
from .refine_net import RefineNet

# Source
from preprocessor import main as preprocessor

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

class EngineCraft:
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

        _, binaryTextMap = cv2.threshold(scoreText, self.textThreshold, 1, 0)
        #_, binaryLinkMap = cv2.threshold(scoreLink, self.linkThreshold, 1, 0)

        #scoreCombined = numpy.clip(binaryTextMap + binaryLinkMap, 0, 1)
        #imageScoreCombined = (scoreCombined * 255).astype(numpy.uint8)
        imageBinaryTextMap = (binaryTextMap * 255).astype(numpy.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        imageEroded = cv2.erode(imageBinaryTextMap.astype(numpy.uint8), kernel)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        imageDilate = cv2.dilate(imageEroded, kernel)

        if self.isDebug:
            preprocessor.write(f"{PATH_ROOT}{PATH_FILE_OUTPUT}craft/{self.uniqueId}/{self.fileName}", "_dilate", imageDilate)

        mergeBoxTollerance = 10
        mergeBoxRowTollerance = 8

        contourList, _ = cv2.findContours(imageDilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxList = []

        for contour in contourList:
            x, y, w, h = cv2.boundingRect(contour)

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

        for a, (x1, y1, w1, h1) in enumerate(boxList):
            isNested = False

            for b, (x2, y2, w2, h2) in enumerate(boxList):
                if a != b:
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
        boxList = self._boxCreation(scoreText, scoreLink, ratioWidth, ratioHeight)

        #fileNameSplit, _ = os.path.splitext(self.fileName)

        #with open(f"{PATH_ROOT}{PATH_FILE_OUTPUT}craft/{self.uniqueId}/{fileNameSplit}.txt", "w") as file:
            #for _, box in enumerate(boxList):
            #    shapeList = numpy.array(box).astype(numpy.int32).reshape((-1))

            #    file.write(",".join([str(shape) for shape in shapeList]) + "\r\n")

            #    cv2.polylines(imageOpen, [shapeList.reshape(-1, 2).reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=1)
        
        for (x, y, w, h) in boxList:
            cv2.rectangle(imageOpen, (x, y), (x + w, y + h), (0, 0, 255), 1)

        if self.isDebug:
            preprocessor.write(f"{PATH_ROOT}{PATH_FILE_OUTPUT}craft/{self.uniqueId}/{self.fileName}", "_result", imageOpen)

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
        
        if self.device == "gpu":
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
        imageOpen = preprocessor.open(f"{PATH_ROOT}{PATH_FILE_INPUT}{self.fileName}")

        targetWidth, targetHeight, ratioWidth, ratioHeight, imageResize, channel = preprocessor.resize(imageOpen, 2048)

        imageGray = preprocessor.gray(imageResize)

        imageBinarize = preprocessor.binarization(imageGray)

        imageNoiseRemove = preprocessor.noiseRemove(imageBinarize)

        imageColor = preprocessor.color(imageNoiseRemove)

        imageResizeCnn = self._resizeCnn(targetWidth, targetHeight, channel, imageColor)

        if self.isDebug:
            preprocessor.write(f"{PATH_ROOT}{PATH_FILE_OUTPUT}craft/{self.uniqueId}/{self.fileName}", "_preprocess", imageColor)

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
            if self.device == "gpu":
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
        if self.device == "gpu":
            detector.load_state_dict(self._removeDataParallel(torch.load(self.pathWeightMain)))

            detector = torch.nn.DataParallel(detector.cuda())
            
            torchBackendCudnn.benchmark = False
        else:
            detector.load_state_dict(self._removeDataParallel(torch.load(self.pathWeightMain, map_location="cpu")))

        detector.eval()

        return detector

    def __init__(self, fileNameValue, deviceValue, isDebugValue, uniqueIdValue):
        self.fileName = fileNameValue
        self.device = deviceValue
        self.isDebug = isDebugValue
        self.uniqueId = uniqueIdValue

        self.pathWeightMain = f"{PATH_ROOT}src/library/engine_craft/mlt_25k.pth"
        self.pathWeightRefine = f"{PATH_ROOT}src/library/engine_craft/refiner_CTW1500.pth"
        self.textThreshold = 0.1
        #self.linkThreshold = 0.1
        self.isRefine = False

        detector = Detector()
        detector = self._detectorEval(detector)

        refineNet = RefineNet()
        refineNet = self._refineNetEval(refineNet)

        imageOpen, ratioWidth, ratioHeight, imageResizeCnn = self._preprocess()

        scoreText, scoreLink = self._inference(imageResizeCnn, detector, refineNet)

        self._result(scoreText, scoreLink, imageOpen, ratioWidth, ratioHeight)