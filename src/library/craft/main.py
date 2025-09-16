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

    ratio, imageResizeCnn, imageRead = helper.preprocess()

    scoreText, scoreLink = helper.inference(imageResizeCnn, detector, refineNet)

    helper.result(scoreText, scoreLink, ratio, imageRead)

Main()