import cv2 as Cv2
import numpy as Numpy
from PIL import Image as PilImage
from PIL import ImageOps as PilImageOps
from PIL import ImageChops as PilImageChops
import easyocr as easyOcr
import os
import sys


def load(fileName):
    return Cv2.imread(f"/home/root/file/input/{fileName}")


def Rescale(image, unit):
    return Cv2.resize(image, None, fx=unit, fy=unit)


def FindText(image, langList, textOutput):
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
        ycenter_ths=0.5,  # 0.5
        height_ths=0.5,  # 0.5
        width_ths=-0.5,  # 0.5
        add_margin=0,  # 0.1
    )

    if textOutput:
        with open("/home/root/file/output/debug.txt", "w") as element:
            for result in resultList:
                print(result, file=element)

    return resultList


def HsvMask(image, colorA, colorB):
    # tmp = Numpy.uint8([[[50, 50, 50]]])
    # hsv = Cv2.cvtColor(tmp, Cv2.COLOR_BGR2HSV)
    # print(hsv)

    imageBgr = Cv2.cvtColor(image, Cv2.COLOR_GRAY2BGR)
    imageHsv = Cv2.cvtColor(imageBgr, Cv2.COLOR_BGR2HSV)

    colorMin = Numpy.array(colorA, Numpy.uint8)
    colorMax = Numpy.array(colorB, Numpy.uint8)

    imageMask = Cv2.inRange(imageHsv, colorMin, colorMax)

    return Cv2.bitwise_and(image, image, mask=imageMask)


def CheckPixelTotal(image, colorA, colorB):
    pixelTotalA = Numpy.sum(image == colorA)
    pixelTotalB = Numpy.sum(image == colorB)

    return [pixelTotalA, pixelTotalB]


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


def CropFix(image):
    imageMask = HsvMask(image, [0, 0, 0], [0, 0, 180])
    pixelPercent = round((imageMask > 0).mean(), 1)
    pixelTotalA, _ = CheckPixelTotal(imageMask, 87, 50)

    image = addBorder(image, (255, 255, 255), 2)

    if (pixelPercent >= 0.5 or pixelPercent == 0.0) and pixelTotalA == 0:
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

    image = load(fileName)
    imageRescale = Rescale(image, 4)
    imageGray = Cv2.cvtColor(imageRescale, Cv2.COLOR_BGR2GRAY)

    imageRectangle = imageGray.copy()
    imageRectangle.fill(255)

    imageResult = imageGray.copy()
    imageResult.fill(255)

    languageCraft = ""
    languageOcr = ""
    psmOcr = "6"

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

    textList = FindText(imageRescale, [languageCraft], False)

    for text in textList:
        coordinate = text[0]
        # label = text[1]

        top = int(coordinate[0][0])
        left = int(coordinate[0][1])
        bottom = int(coordinate[2][0])
        right = int(coordinate[2][1])

        """imageRectangle = Cv2.rectangle(
            imageGray, (top, left), (bottom, right), (0, 0, 0), 1
        )"""

        imageCrop = imageGray[left:right, top:bottom]
        imageCropFix = CropFix(imageCrop)

        imageResult[left:right, top:bottom] = imageCropFix

    """imagePil = PilImage.fromarray(imageRectangle)
    imagePil.save(f"/home/root/file/output/{fileName}_box.png", dpi=(300, 300))"""

    imagePil = PilImage.fromarray(imageResult).convert("1", dither=PilImage.Dither.NONE)
    imagePil.save(f"/home/root/file/output/{fileName}", dpi=(300, 300))

    os.system(
        f"tesseract '/home/root/file/output/{fileName}' '/home/root/file/output/{fileName}' -l {languageOcr} --oem 3 --psm {psmOcr} -c preserve_interword_spaces=1 -c tessedit_char_blacklist='〇,' txt"
    )


# TO DO - Integrate dewrap
Main()
