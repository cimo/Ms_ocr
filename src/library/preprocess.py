import cv2 as Cv2
import numpy as Numpy
from PIL import Image as PilImage
import easyocr as easyOcr
import os
import sys
import warnings

# FutureWarning: You are using `torch.load` with `weights_only=False
warnings.filterwarnings("ignore", category=FutureWarning)


def load(fileName):
    result = False

    path = f"/home/app/file/input/{fileName}"

    if os.path.isfile(path) and os.access(path, os.R_OK):
        image = Cv2.imread(path)

        if len(image.shape) == 2:
            image = Cv2.cvtColor(image, Cv2.COLOR_GRAY2BGR)
        if image.shape[0] == 2:
            image = image[0]
        if image.shape[2] == 4:
            image = image[:, :, :3]

        result = Numpy.array(image)

    return result


def Rescale(image, unit):
    imageHeight, imageWidth, _ = image.shape

    size = unit * max(imageHeight, imageWidth)

    if size > 4096:
        size = 4096

    ratio = size / max(imageHeight, imageWidth)

    imageHeightTarget, imageWidthTarget = int(imageHeight * ratio), int(
        imageWidth * ratio
    )

    return Cv2.resize(image, (imageWidthTarget, imageHeightTarget), interpolation=None)


def FindText(image, langList, debug):
    easyOcrReader = easyOcr.Reader(langList, gpu=False)
    resultList = easyOcrReader.readtext(
        image,
        detail=1,  # 1
        paragraph=False,  # False
        contrast_ths=0.1,  # 0.1
        adjust_contrast=0.5,  # 0.5
        text_threshold=0.7,  # 0.7
        low_text=0.4,  # 0.4
        link_threshold=0.4,  # 0.4
        mag_ratio=1,  # 1
        slope_ths=0.1,  # 0.1
        ycenter_ths=0.8,  # 0.5
        height_ths=0.5,  # 0.5
        width_ths=-0.1,  # 0.5
        add_margin=0.0,  # 0.1
    )

    if debug == "true":
        with open("/home/app/file/output/debug.txt", "w") as element:
            for result in resultList:
                print(result, file=element)

    return resultList


def addBorder(image, color, unit):
    return Cv2.rectangle(image, (0, image.shape[0]), (image.shape[1], 0), color, unit)


def Binarization(image, unitBlur, unitThresholdA, unitThresholdB, isInverted):
    thresholdBinary = Cv2.THRESH_BINARY

    if isInverted:
        thresholdBinary = Cv2.THRESH_BINARY_INV

    imageBlur = Cv2.GaussianBlur(image, (unitBlur, unitBlur), 0)
    imageDivide = Cv2.divide(image, imageBlur, scale=255)

    return Cv2.adaptiveThreshold(
        imageDivide,
        255,
        Cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdBinary,
        unitThresholdA,
        unitThresholdB,
    )


def backgroundCheck(image):
    imageBgr = Cv2.cvtColor(image, Cv2.COLOR_GRAY2BGR)
    imageHsv = Cv2.cvtColor(imageBgr, Cv2.COLOR_BGR2HSV)

    colorMin = Numpy.array([0, 0, 0])
    colorMax = Numpy.array([179, 255, 146])
    mask = Cv2.inRange(imageHsv, colorMin, colorMax)

    kernel = Cv2.getStructuringElement(Cv2.MORPH_RECT, (5, 3))
    dilatation = Cv2.dilate(mask, kernel, iterations=5)

    return 255 - Cv2.bitwise_and(dilatation, mask)


def CropFix(image):
    backgroundCheckResult = backgroundCheck(image)
    pixelQuantity = round((backgroundCheckResult > 0).mean(), 1)

    image = addBorder(image, (255, 255, 255), 2)

    if pixelQuantity == 1.0 or pixelQuantity <= 0.4:
        imageThreshold = Binarization(image, 91, 255, 0, True)

        image = addBorder(imageThreshold, (255, 255, 255), 3)
    else:
        image = Binarization(image, 91, 255, 5, False)

    image = NoiseRemove(image, 1)

    return image


def NoiseRemove(image, unit):
    kernel = Numpy.ones((unit, unit), Numpy.uint8)

    return Cv2.morphologyEx(image, Cv2.MORPH_CLOSE, kernel)


def FontUp(image, unit):
    kernel = Numpy.ones((2, 2), Numpy.uint8)

    image = Cv2.bitwise_not(image)
    image = Cv2.dilate(image, kernel, iterations=unit)

    return Cv2.bitwise_not(image)


def FontDown(image, unit):
    kernel = Numpy.ones((2, 2), Numpy.uint8)

    image = Cv2.bitwise_not(image)
    image = Cv2.erode(image, kernel, iterations=unit)

    return Cv2.bitwise_not(image)


def Main():
    fileName = sys.argv[1]
    language = sys.argv[2]
    result = sys.argv[3]
    debug = sys.argv[4]

    image = load(fileName)

    if not isinstance(image, bool):
        imageRescale = Rescale(image, 4)
        imageGray = Cv2.cvtColor(imageRescale, Cv2.COLOR_BGR2GRAY)
        imageRectangle = imageGray.copy()

        imageResult = imageGray.copy()
        imageResult.fill(255)

        languageCraft = ""
        languageOcr = ""
        psmOcr = 6

        if language == "en":
            languageCraft = "en"
            languageOcr = "eng"
        elif language == "jp":
            languageCraft = "ja"
            languageOcr = "Japanese"
        elif language == "jp_vert":
            languageCraft = "ja"
            languageOcr = "Japanese_vert"
            psmOcr = 3

        textList = FindText(imageRescale, [languageCraft], debug)

        for text in textList:
            coordinate = text[0]
            # label = text[1]

            top = int(coordinate[0][0])
            left = int(coordinate[0][1])
            bottom = int(coordinate[2][0])
            right = int(coordinate[2][1])

            if debug == "true":
                Cv2.rectangle(
                    imageRectangle, (top, left), (bottom, right), (0, 0, 0), 1
                )

            imageCrop = imageGray[left:right, top:bottom]
            imageCropFix = CropFix(imageCrop)

            imageResult[left:right, top:bottom] = imageCropFix

        if debug == "true":
            Cv2.imwrite(f"/home/app/file/output/{fileName}_box.png", imageRectangle)

        imagePil = PilImage.fromarray(imageResult).convert(
            "1", dither=PilImage.Dither.NONE
        )
        imagePil.save(f"/home/app/file/output/{fileName}_result.png", dpi=(300, 300))

        os.system(
            f"tesseract '/home/app/file/output/{fileName}_result.png' '/home/app/file/output/{fileName}' -l {languageOcr} --oem 1 --psm {psmOcr} -c preserve_interword_spaces=1 -c page_separator='' -c tessedit_char_blacklist='〇,' {result}"
        )
    else:
        print(f"File not exists!")


# TO DO - Create craft
# TO DO - Integrate dewarp
Main()
