import os
import time
import cv2
import numpy
import torch
from torch.autograd import Variable as torchAutograd
from collections import OrderedDict as collectionOrderDict

#Source
import craft_utils

pathRoot = os.path.dirname(os.path.abspath(__file__))
pathInput = os.path.join(pathRoot, "input")
pathOutput = os.path.join(pathRoot, "output")

imageName = "test_5.jpg"
modelMain = "craft_mlt_25k.pth"
modelRefine="craft_refiner_CTW1500.pth"
lowText = 0.4
thresholdText = 0.7
thresholdLink = 0.4
canvasSize = 4096
magRatio = 1.0
isCuda=False
isRefine=True
isPoly=False

def _imageResize(image):
    height, width, channel = image.shape

    size = magRatio * max(width, height)

    if size > canvasSize:
        size = canvasSize
    
    ratio = size / max(width, height)
    
    widthTarget = int(width * ratio)
    heightTarget = int(height * ratio)

    imageResize = cv2.resize(image, (widthTarget, heightTarget), interpolation=cv2.INTER_LINEAR)

    widthTarget32 = widthTarget
    heightTarget32 = heightTarget
    
    if heightTarget % 32 != 0:
        heightTarget32 = heightTarget + (32 - heightTarget % 32)
    
    if widthTarget % 32 != 0:
        widthTarget32 = widthTarget + (32 - widthTarget % 32)
    
    imageResult = numpy.zeros((heightTarget32, widthTarget32, channel), dtype=numpy.float32)
    imageResult[0:heightTarget, 0:widthTarget, :] = imageResize

    return ratio, imageResult

def _normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(numpy.float32)
    img -= numpy.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=numpy.float32)
    img /= numpy.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=numpy.float32)
    return img

def _denormalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy()
    img *= variance
    img += mean
    img *= 255.0
    img = numpy.clip(img, 0, 255).astype(numpy.uint8)
    return img

def removeDataParallel(stateDict):
    if list(stateDict.keys())[0].startswith("module"):
        indexStart = 1
    else:
        indexStart = 0

    stateDictNew = collectionOrderDict()

    for key, value in stateDict.items():
        name = ".".join(key.split(".")[indexStart:])

        stateDictNew[name] = value

    return stateDictNew

def preprocess(image):
    os.makedirs(pathOutput, exist_ok=True)

    imageLoad = cv2.imread(image)

    if len(imageLoad.shape) == 2:
        imageLoad = cv2.cvtColor(imageLoad, cv2.COLOR_GRAY2BGR)
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
    imageMorphology = cv2.morphologyEx(imageBinarization, cv2.MORPH_CLOSE, kernel)
    imageColor = cv2.cvtColor(imageMorphology, cv2.COLOR_GRAY2BGR)

    ratio, imageResize = _imageResize(imageColor)
    
    ratioW = 1 / ratio
    ratioH = 1 / ratio

    fileName, fileExtension = os.path.splitext(os.path.basename(f"{pathInput}/{imageName}"))
    path = os.path.join(pathOutput, f"{fileName}_preprocess{fileExtension}")
    imageWrite = (numpy.clip(imageResize, 0, 1) * 255).astype(numpy.uint8)
    cv2.imwrite(path, imageWrite)

    return imageResize, ratioW, ratioH

def inference(imageValue, craft, refineNet):
    timeStart = time.time()

    image = numpy.array(imageValue)
    imageNormalize  = _normalizeMeanVariance(image)
    imageTensor = torch.from_numpy(imageNormalize).permute(2, 0, 1)
    modelInput = torchAutograd(imageTensor.unsqueeze(0))
    
    if isCuda:
        modelInput = modelInput.cuda()

    with torch.no_grad():
        scoreMap, feature = craft(modelInput)

    scoreTextTensor = scoreMap[0,:,:,0]
    scoreLinkTensor = scoreMap[0,:,:,1]

    scoreText = scoreTextTensor.cpu().numpy()
    scoreLink = scoreLinkTensor.cpu().numpy()

    if refineNet is not None:
        with torch.no_grad():
            refineMap = refineNet(scoreMap, feature)
        
        scoreLinkTensor = refineMap[0,:,:,0]
        scoreLink = scoreLinkTensor.cpu().numpy()
    
    print(f"Time inference: {time.time() - timeStart}")

    return scoreText, scoreLink

def postprocess(scoreText, scoreLink, ratioW, ratioH):
    box, polyList = craft_utils.getDetBoxes(scoreText, scoreLink, thresholdText, thresholdLink, lowText, isPoly)
    box = craft_utils.adjustResultCoordinates(box, ratioW, ratioH)
    polyList = craft_utils.adjustResultCoordinates(polyList, ratioW, ratioH)
    
    for index in range(len(polyList)):
        if polyList[index] is None: polyList[index] = box[index]

    scoreTextCopy = scoreText.copy()
    hStack = numpy.hstack((scoreTextCopy, scoreLink))
    
    fileName, fileExtension = os.path.splitext(os.path.basename(f"{pathInput}/{imageName}"))
    path = os.path.join(pathOutput, f"{fileName}_mask{fileExtension}")
    imageWrite = (numpy.clip(hStack, 0, 1) * 255).astype(numpy.uint8)
    imageMask = cv2.applyColorMap(imageWrite, cv2.COLORMAP_JET)
    cv2.imwrite(path, imageMask)

    return box, polyList

def output(polyList, image):
    fileName, fileExtension = os.path.splitext(os.path.basename(f"{pathInput}/{imageName}"))
    pathText = os.path.join(pathOutput, f"{fileName}.txt")
    pathImage = os.path.join(pathOutput, f"{fileName}{fileExtension}")

    with open(pathText, "w") as file:
        for _, poly in enumerate(polyList):
            shapeList = numpy.array(poly).astype(numpy.int32).reshape((-1))

            file.write(",".join([str(shape) for shape in shapeList]) + "\r\n")

            cv2.polylines(image, [shapeList.reshape(-1, 2).reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=1)

    imageWrite = (numpy.clip(image, 0, 1) * 255).astype(numpy.uint8)
    cv2.imwrite(pathImage, imageWrite)
