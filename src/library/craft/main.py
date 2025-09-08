# Source
import helper
from craft import Craft
from refine_net import RefineNet

def Main():
    print(f"On your machine CUDA are: {'available' if helper.torch.cuda.is_available() else 'NOT available'}.")

    craft = Craft(False, False)
    craft = helper.craftEval(craft)

    refineNet = RefineNet()
    refineNet = helper.refineNetEval(refineNet)

    print(f"Load file: {helper.pathRoot}{helper.pathInput}{helper.fileName}\r")

    image, imageResize, ratioWidth, ratioHeight = helper.preprocess()

    scoreText, scoreLink = helper.inference(imageResize, craft, refineNet)

    helper.result(scoreText, scoreLink, ratioWidth, ratioHeight, image)

Main()