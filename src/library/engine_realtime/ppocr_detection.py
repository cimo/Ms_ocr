import cv2
import numpy

class PPocrDetection:
    @property
    def name(self):
        return self.__class__.__name__

    def setInputSize(self, input_size):
        self.inputSize = tuple(input_size)
        self.model.setInputSize(self.inputSize)
        self.model.setInputMean((123.675, 116.28, 103.53))
        self.model.setInputScale(1.0/255.0/numpy.array([0.229, 0.224, 0.225]))

    def setBackendAndTarget(self, backendId, targetId):
        self.backendId = backendId
        self.targetId = targetId
        self.model.setPreferableBackend(self.backendId)
        self.model.setPreferableTarget(self.targetId)

    def infer(self, image):
        assert image.shape[0] == self.inputSize[1], '{} (height of input image) != {} (preset height)'.format(image.shape[0], self.inputSize[1])
        assert image.shape[1] == self.inputSize[0], '{} (width of input image) != {} (preset width)'.format(image.shape[1], self.inputSize[0])

        return self.model.detect(image)

    def __init__(self, modelPath, inputSize=[736, 736], binaryThreshold=0.3, polygonThreshold=0.5, maxCandidates=200, unclipRatio=2.0, backendId=0, targetId=0):
        self.modelPath = modelPath
        self.model = cv2.dnn_TextDetectionModel_DB(cv2.dnn.readNet(self.modelPath))

        self.inputSize = tuple(inputSize)
        self.inputHeight = inputSize[0]
        self.inputWidth = inputSize[1]
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
