import os
import time
import cv2
import numpy
import torch
import torch.backends.cudnn as torchBackendCudnn
from torch.autograd import Variable as torchAutograd
from collections import OrderedDict as collectionOrderDict

# Source
from craft import Craft
from refine_net import RefineNet
import craft_utils
import imgproc
import file_utils

pathRoot = os.path.dirname(os.path.abspath(__file__))
pathInput = os.path.join(pathRoot, "input")
pathOutput = os.path.join(pathRoot, "output")
imageName = "test_5.jpg"

trainedModel = "craft_mlt_25k.pth"
lowText = 0.4
thresholdText = 0.7
thresholdLink = 0.4
canvasSize = 4096
magRatio = 1.0
cuda=False
poly=False
refine=True
refineModel="craft_refiner_CTW1500.pth"

'''
def merge_boxes_by_distance(polys: list, max_dx: int, max_dy: int) -> list:
    if len(polys) == 0:
        return []

    # Converti poligoni in [x_min, y_min, x_max, y_max]
    rects = []
    for p in polys:
        x_min = np.min(p[:,0])
        y_min = np.min(p[:,1])
        x_max = np.max(p[:,0])
        y_max = np.max(p[:,1])
        rects.append([x_min, y_min, x_max, y_max])
    rects = np.array(rects)

    merged = []
    used = [False] * len(rects)

    for i in range(len(rects)):
        if used[i]:
            continue
        # inizia un nuovo gruppo
        group = [rects[i]]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j in range(len(rects)):
                if used[j]:
                    continue
                # distanza minima tra il box j e qualsiasi box del gruppo
                for b in group:
                    dx = max(0, max(b[0], rects[j][0]) - min(b[2], rects[j][2]))
                    dy = max(0, max(b[1], rects[j][1]) - min(b[3], rects[j][3]))
                    if dx <= max_dx and dy <= max_dy:
                        group.append(rects[j])
                        used[j] = True
                        changed = True
                        break
        # crea bounding box rettangolare del gruppo
        group = np.array(group)
        x0 = group[:,0].min()
        y0 = group[:,1].min()
        x1 = group[:,2].max()
        y1 = group[:,3].max()
        merged.append(np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]]))
    return merged
'''

def _preprocess(image):
    imageLoad = cv2.imread(image)

    if len(imageLoad.shape) == 2:
        imageLoad = cv2.cvtColor(imageLoad, cv2.COLOR_GRAY2BGR)
    if imageLoad.shape[0] == 2:
        imageLoad = imageLoad[0]
    if imageLoad.shape[2] == 4:
        imageLoad = imageLoad[:, :, :3]

    imageGray = cv2.cvtColor(imageLoad, cv2.COLOR_BGR2GRAY)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #imageContrast = clahe.apply(imageGray)
    imageBlur = cv2.GaussianBlur(imageGray, (91, 91), 0)
    imageDivide = cv2.divide(imageGray, imageBlur, scale=255)

    imageBinarization = cv2.adaptiveThreshold(
        imageDivide,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        255,
        0
    )

    kernel = numpy.ones((1, 1), numpy.uint8)
    imageFinal = cv2.morphologyEx(imageBinarization, cv2.MORPH_CLOSE, kernel)

    fileName, fileExtension = os.path.splitext(os.path.basename(f"{pathInput}/{imageName}"))
    pathPreprocess = os.path.join(pathOutput, f"{fileName}_preprocess{fileExtension}")
    cv2.imwrite(pathPreprocess, imageFinal)

    result = cv2.cvtColor(imageFinal, cv2.COLOR_GRAY2BGR)

    return numpy.array(result)

def _execute(craft, image, poly, refineNet):
    # Inference
    time0 = time.time()

    imageResized, ratioTarget, _ = imgproc.resize_aspect_ratio(image, canvasSize, interpolation=cv2.INTER_LINEAR, mag_ratio=magRatio)
    
    ratioH = ratioW = 1 / ratioTarget

    x = imgproc.normalizeMeanVariance(imageResized)
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = torchAutograd(x.unsqueeze(0))
    
    if cuda:
        x = x.cuda()

    with torch.no_grad():
        y, feature = craft(x)

    scoreText = y[0,:,:,0].cpu().data.numpy()
    scoreLink = y[0,:,:,1].cpu().data.numpy()

    if refineNet is not None:
        with torch.no_grad():
            refineY = refineNet(y, feature)
        
        scoreLink = refineY[0,:,:,0].cpu().data.numpy()

    time0 = time.time() - time0
    
    # Post process
    time1 = time.time()

    box, polyList = craft_utils.getDetBoxes(scoreText, scoreLink, thresholdText, thresholdLink, lowText, poly)

    box = craft_utils.adjustResultCoordinates(box, ratioW, ratioH)
    polyList = craft_utils.adjustResultCoordinates(polyList, ratioW, ratioH)
    
    for index in range(len(polyList)):
        if polyList[index] is None: polyList[index] = box[index]

    time1 = time.time() - time1

    scoreTextImage = scoreText.copy()
    scoreTextImage = numpy.hstack((scoreTextImage, scoreLink))
    scoreTextImage = (numpy.clip(scoreTextImage, 0, 1) * 255).astype(numpy.uint8)
    scoreTextResult = cv2.applyColorMap(scoreTextImage, cv2.COLORMAP_JET)

    print(f"Time (inference / post process): {time0} / {time1}")

    return box, polyList, scoreTextResult

def _removeDataParallelPrefix(stateDict):
    if list(stateDict.keys())[0].startswith("module"):
        indexStart = 1
    else:
        indexStart = 0

    stateDictNew = collectionOrderDict()

    for key, value in stateDict.items():
        name = ".".join(key.split(".")[indexStart:])

        stateDictNew[name] = value

    return stateDictNew

if __name__ == "__main__":
    craft = Craft()

    print(f"Loading weight: {trainedModel}")

    if cuda:
        craft.load_state_dict(_removeDataParallelPrefix(torch.load(trainedModel)))
    else:
        craft.load_state_dict(_removeDataParallelPrefix(torch.load(trainedModel, map_location="cpu")))

    if cuda:
        craft = torch.nn.DataParallel(craft.cuda())
        
        torchBackendCudnn.benchmark = False

    craft.eval()

    refineNet = None

    if refine:
        refineNet = RefineNet()

        print(f"Loading refiner weight: {refineModel}")

        if cuda:
            refineNet.load_state_dict(_removeDataParallelPrefix(torch.load(refineModel)))
            refineNet = torch.nn.DataParallel(refineNet.cuda())
        else:
            refineNet.load_state_dict(_removeDataParallelPrefix(torch.load(refineModel, map_location="cpu")))

        refineNet.eval()

    print(f"Image: {pathInput}/{imageName}\r")
    
    image = _preprocess(f"{pathInput}/{imageName}")

    _, polyList, scoreText = _execute(craft, image, poly, refineNet)

    #merged_polys = merge_boxes_by_distance(polyList, max_dx=0.1, max_dy=0.1)
    #merged_polys = merge_boxes_by_distance(polyList, max_dx=3, max_dy=0.1)

    file_utils.saveResult(f"{pathInput}/{imageName}", image[:,:,::-1], polyList, dirname=pathOutput)

    fileName, fileExtension = os.path.splitext(os.path.basename(f"{pathInput}/{imageName}"))
    pathMask = os.path.join(pathOutput, f"{fileName}_mask{fileExtension}")
    cv2.imwrite(pathMask, scoreText)