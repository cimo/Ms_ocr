import cv2
import numpy

# Soruce
from .charset_crnn import CHARSET_EN_36, CHARSET_CH_94, CHARSET_CN_3944

class RecognizerCrnn:
    @property
    def name(self):
        return self.__class__.__name__

    def _load_charset(self, charset):
        return ''.join(charset.splitlines())

    def _preprocess(self, image, rbbox):
        vertices = rbbox.reshape((4, 2)).astype(numpy.float32)

        rotationMatrix = cv2.getPerspectiveTransform(vertices, self._targetVertices)
        cropped = cv2.warpPerspective(image, rotationMatrix, (self._inputWidth, self._inputHeight))

        if 'CN' in self._model_path or 'CH' in self._model_path:
            return cv2.dnn.blobFromImage(
                cropped,
                scalefactor=1.0 / 127.5,
                size=(self._inputWidth, self._inputHeight),
                mean=127.5,
                swapRB=True,
                crop=False
            )
        else:
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            
            return cv2.dnn.blobFromImage(
                cropped,
                scalefactor=1.0 / 127.5,
                size=(self._inputWidth, self._inputHeight),
                mean=127.5,
                swapRB=False,
                crop=False
            )

    def _postprocess(self, outputBlob):
        text = ''

        for i in range(outputBlob.shape[0]):
            c = numpy.argmax(outputBlob[i][0])

            if c != 0:
                text += self._charset[c - 1]
            else:
                text += '-'

        char_list = []

        for i in range(len(text)):
            if text[i] != '-' and (not (i > 0 and text[i] == text[i - 1])):
                char_list.append(text[i])

        return ''.join(char_list)

    def __init__(self, modelPath, backendId=0, targetId=0):
        self._model_path = modelPath
        self._backendId = backendId
        self._targetId = targetId

        self._model = cv2.dnn.readNet(self._model_path)
        self._model.setPreferableBackend(self._backendId)
        self._model.setPreferableTarget(self._targetId)

        if '_EN_' in self._model_path:
            self._charset = self._load_charset(CHARSET_EN_36)
        elif '_CH_' in self._model_path:
            self._charset = self._load_charset(CHARSET_CH_94)
        elif '_CN_' in self._model_path:
            self._charset = self._load_charset(CHARSET_CN_3944)
        else:
            print('Charset not supported! Exiting ...')

            exit()

        self._inputSize = (100, 32)
        self._inputWidth, self._inputHeight = self._inputSize
        self._targetVertices = numpy.array([
            [0, self._inputHeight - 1],
            [0, 0],
            [self._inputWidth - 1, 0],
            [self._inputWidth - 1, self._inputHeight - 1]
        ], dtype=numpy.float32)

    def setBackendAndTarget(self, backendId, targetId):
        self._backendId = backendId
        self._targetId = targetId
        self._model.setPreferableBackend(self._backendId)
        self._model.setPreferableTarget(self._targetId)

    def infer(self, image, rbbox):
        inputBlob = self._preprocess(image, rbbox)

        self._model.setInput(inputBlob)
        outputBlob = self._model.forward()

        results = self._postprocess(outputBlob)

        return results
