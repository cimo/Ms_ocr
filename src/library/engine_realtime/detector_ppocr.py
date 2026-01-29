import cv2
import numpy

class DetectorPPocr:
    @property
    def name(self):
        return self.__class__.__name__

    def setInputSize(self, input_size):
        self.inputSize = (int(input_size[0]), int(input_size[1]))
        self.inputWidth, self.inputHeight = self.inputSize
        self.model.setInputSize(self.inputSize)
        self.model.setInputMean((123.675, 116.28, 103.53))
        self.model.setInputScale(1.0/255.0/numpy.array([0.229, 0.224, 0.225]))

    def setBackendAndTarget(self, backendId, targetId):
        self.backendId = backendId
        self.targetId = targetId
        self.model.setPreferableBackend(self.backendId)
        self.model.setPreferableTarget(self.targetId)

    def infer(self, image):
        assert image.shape[0] == self.inputHeight, '{} (height of input image) != {} (preset height)'.format(image.shape[0], self.inputHeight)
        assert image.shape[1] == self.inputWidth,  '{} (width of input image) != {} (preset width)'.format(image.shape[1], self.inputWidth)

        return self.model.detect(image)

    def __init__(self, modelPath, inputSize=[736, 736], binaryThreshold=0.3, polygonThreshold=0.5, maxCandidates=200, unclipRatio=2.0, backendId=0, targetId=0):
        self.modelPath = modelPath
        self.model = cv2.dnn_TextDetectionModel_DB(cv2.dnn.readNet(self.modelPath))

        self.inputSize = (int(inputSize[0]), int(inputSize[1]))
        self.inputWidth, self.inputHeight = self.inputSize
        self.binaryThreshold = binaryThreshold
        self.polygonThreshold = polygonThreshold
        self.maxCandidates = maxCandidates
        self.unclipRatio = unclipRatio
        self.backendId = backendId
        self.targetId = targetId

        self.model.setPreferableBackend(self.backendId)
        self.model.setPreferableTarget(self.targetId)

        self.model.setBinaryThreshold(self.binaryThreshold)
        self.model.setPolygonThreshold(self.polygonThreshold)
        self.model.setUnclipRatio(self.unclipRatio)
        self.model.setMaxCandidates(self.maxCandidates)

        self.model.setInputSize(self.inputSize)
        self.model.setInputMean((123.675, 116.28, 103.53))
        self.model.setInputScale(1.0/255.0/numpy.array([0.229, 0.224, 0.225]))
        self.model.setInputSwapRB(True)
