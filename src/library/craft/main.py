# Source
import helper
from detector import Detector
from refine_net import RefineNet

def Main():
    helper.checkCuda()

    detector = Detector(False, False)
    detector = helper.detectorEval(detector)

    refineNet = RefineNet()
    refineNet = helper.refineNetEval(refineNet)

    image, imageResize, ratio = helper.preprocess()

    scoreText, scoreLink = helper.inference(imageResize, detector, refineNet)

    helper.result(scoreText, scoreLink, ratio, image)

Main()