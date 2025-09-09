# Source
import helper
from craft import Craft
from refine_net import RefineNet

def Main():
    helper.checkCuda()

    craft = Craft(False, False)
    craft = helper.craftEval(craft)

    refineNet = RefineNet()
    refineNet = helper.refineNetEval(refineNet)

    image, imageResize, ratioWidth, ratioHeight = helper.preprocess()

    scoreText, scoreLink = helper.inference(imageResize, craft, refineNet)

    helper.result(scoreText, scoreLink, ratioWidth, ratioHeight, image)

Main()