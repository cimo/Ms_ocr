# Source
import helper

def Main():
    pathInput = "/home/app/file/input/1_jp.jpg"
    pathOutput = "/home/app/file/output/preprocessor/1_jp.jpg"

    image = helper.open(pathInput)

    _, _, _, imageResize, _ = helper.resize(image, 2048)
    helper.write(pathOutput, "_resize", imageResize)

    _, _, _, imageResizeLineHeight, _ = helper.resizeLineHeight(image)
    helper.write(pathOutput, "_resizeLineHeight", imageResizeLineHeight)

    imageGray = helper.gray(imageResizeLineHeight)
    helper.write(pathOutput, "_gray", imageGray)

    imageColor = helper.color(imageGray)
    helper.write(pathOutput, "_color", imageColor)

    imageMedianBlur = helper.medianBlur(imageGray)
    helper.write(pathOutput, "_medianBlur", imageMedianBlur)

    imageBinarization = helper.binarization(imageGray)
    helper.write(pathOutput, "_binarization", imageBinarization)

    contourList, hierarchy = helper.contour(imageGray)
    print(f"contourList: {contourList} - hierarchy: {hierarchy}")

    imageMask = helper.mask(imageGray)
    helper.write(pathOutput, "_mask", imageMask)

    imageMaskApply = helper.maskApply(imageGray, imageMask)
    helper.write(pathOutput, "_maskApply", imageMaskApply)

    imageGamma = helper.gamma(imageGray)
    helper.write(pathOutput, "_gamma", imageGamma)

    imageNoiseRemove = helper.noiseRemove(imageGray)
    helper.write(pathOutput, "_noiseRemove", imageNoiseRemove)

    imageBorder = helper.addOrRemoveBorder(imageGray)
    helper.write(pathOutput, "_border", imageBorder)

    imageHeatmap1 = helper.heatmap(helper.numpy.random.rand(256, 256))
    imageHeatmap2 = helper.heatmap(helper.numpy.random.rand(256, 256))
    helper.write(pathOutput, "_heatmap1", imageHeatmap1)
    helper.write(pathOutput, "_heatmap2", imageHeatmap2)

Main()