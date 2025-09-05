import torch
import torch.backends.cudnn as torchBackendCudnn

# Source
import helper
from craft import Craft
from refine_net import RefineNet

if __name__ == "__main__":
    print(f"On your machine CUDA are: {'available' if torch.cuda.is_available() else 'NOT available'}.")

    craft = Craft(False, False)

    print(f"Loading weight: {helper.weightMain}")

    if helper.isCuda:
        craft.load_state_dict(helper.removeDataParallel(torch.load(helper.weightMain)))

        craft = torch.nn.DataParallel(craft.cuda())
        
        torchBackendCudnn.benchmark = False
    else:
        craft.load_state_dict(helper.removeDataParallel(torch.load(helper.weightMain, map_location="cpu")))

    craft.eval()

    refineNet = None

    if helper.isRefine:
        refineNet = RefineNet()

        print(f"Loading weight refineNet: {helper.weightRefine}")

        if helper.isCuda:
            refineNet.load_state_dict(helper.removeDataParallel(torch.load(helper.weightRefine)))

            refineNet = torch.nn.DataParallel(refineNet.cuda())

            torchBackendCudnn.benchmark = False
        else:
            refineNet.load_state_dict(helper.removeDataParallel(torch.load(helper.weightRefine, map_location="cpu")))

        refineNet.eval()

    print(f"Image: {helper.pathInput}/{helper.imageName}\r")

    image, imageResize, ratioWidth, ratioHeight = helper.preprocess(f"{helper.pathInput}/{helper.imageName}")

    scoreText, scoreLink = helper.inference(imageResize, craft, refineNet)

    helper.output(scoreText, scoreLink, ratioWidth, ratioHeight, image)