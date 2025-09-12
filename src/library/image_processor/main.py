# Source
import helper

def Main():
    path = "/home/app/file/input/1_jp.jpg"

    image = helper.read(path)

    imageGray = helper.gray(image)

    helper.write("/home/app/file/input/1_jp.jpg", "_gray", imageGray)

    imageResize, _ = helper.resize(image, 1024)

    helper.write("/home/app/file/input/1_jp.jpg", "_resize", imageResize)

    imageBinarization = helper.binarization(imageGray)

    helper.write("/home/app/file/input/1_jp.jpg", "_binarization", imageBinarization)

    imageGamma = helper.gamma(image)

    helper.write("/home/app/file/input/1_jp.jpg", "_gamma", imageGamma)

    imageNoiseRemove = helper.noiseRemove(imageGray, modeValue="close", unit=1)

    helper.write("/home/app/file/input/1_jp.jpg", "_noiseRemove", imageNoiseRemove)

    imageBorder = helper.addOrRemoveBorder(image)

    helper.write("/home/app/file/input/1_jp.jpg", "_border", imageBorder)

Main()