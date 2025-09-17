# Source
import helper

def Main():
    pathInput = "/home/app/file/input/1_jp.jpg"
    pathOutput = "/home/app/file/output/preprocessor/1_jp.jpg"

    image = helper.read(pathInput)

    _, _, _, imageResize, _ = helper.resize(image, 1024)
    helper.write(pathOutput, "_resize", imageResize)

    _, imageResizeLineHeight = helper.resizeLineHeight(image)
    helper.write(pathOutput, "_resizeLineHeight", imageResizeLineHeight)

    imageGray = helper.gray(imageResize)
    helper.write(pathOutput, "_gray", imageGray)

    imageColor = helper.color(imageGray)
    helper.write(pathOutput, "_color", imageColor)

    imageBinarization = helper.binarization(imageGray)
    helper.write(pathOutput, "_binarization", imageBinarization)

    imageGamma = helper.gamma(imageGray)
    helper.write(pathOutput, "_gamma", imageGamma)

    imageNoiseRemove = helper.noiseRemove(imageGray)
    helper.write(pathOutput, "_noiseRemove", imageNoiseRemove)

    imageBorder = helper.addOrRemoveBorder(imageGray)
    helper.write(pathOutput, "_border", imageBorder)

    imageHeatmap = helper.heatmap(helper.numpy.random.rand(256, 256), helper.numpy.random.rand(256, 256))
    helper.write(pathOutput, "_heatmap", imageHeatmap)

Main()