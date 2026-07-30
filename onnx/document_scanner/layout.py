import sys
sys.dont_write_bytecode = True

import os
import cv2
import numpy

sys.path.append(f"{os.path.dirname(__file__)}/..")
from helper import onnxSessionBuild

class Layout:
    def execute(self, imageRgb):
        resultList = []

        imageHeight, imageWidth = imageRgb.shape[0:2]
        imageResized = cv2.resize(imageRgb, (800, 800), interpolation=cv2.INTER_CUBIC).astype(numpy.float32) / 255.0

        tensor = numpy.expand_dims(imageResized.transpose((2, 0, 1)), axis=0).astype(numpy.float32)

        tensorFeedObject = {
            "image": tensor,
            "im_shape": numpy.array([[800, 800]], dtype=numpy.float32),
            "scale_factor": numpy.array([[800 / float(imageHeight), 800 / float(imageWidth)]], dtype=numpy.float32)
        }

        tensorOutputList = self.onnxSession.run(None, tensorFeedObject)

        boxCount = int(tensorOutputList[1][0]) if len(tensorOutputList) > 1 else len(tensorOutputList[0])

        for a in range(boxCount):
            value = tensorOutputList[0][a]

            classId = int(value[0])
            score = float(value[1])
            x1 = max(0, min(int(round(float(value[2]))), imageWidth))
            y1 = max(0, min(int(round(float(value[3]))), imageHeight))
            x2 = max(0, min(int(round(float(value[4]))), imageWidth))
            y2 = max(0, min(int(round(float(value[5]))), imageHeight))

            if score >= self.scoreThreshold and x2 > x1 and y2 > y1:
                label = self.labelObject[classId] if classId in self.labelObject else str(classId)

                resultList.append({
                    "label": label,
                    "score": score,
                    "coordinate": [x1, y1, x2, y2]
                })

        return resultList

    def __init__(self):
        self.osPathDirName = f"{os.path.dirname(__file__)}/"
        self.pathModel = f"{self.osPathDirName}model/pp-docLayout_plus-l.onnx"

        self.scoreThreshold = 0.3

        self.labelObject = {
            12: "header",
            10: "doc_title",
            4: "abstract",
            5: "content",
            0: "paragraph_title",
            2: "text",
            1: "image",
            6: "figure_title",
            16: "chart",
            8: "table",
            7: "formula",
            17: "formula_number",
            13: "algorithm",
            18: "aside_text",
            9: "reference",
            19: "reference_content",
            11: "footnote",
            14: "footer",
            3: "number",
            15: "seal"
        }

        self.onnxSession = onnxSessionBuild(self.pathModel)
