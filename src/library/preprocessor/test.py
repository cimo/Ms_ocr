# Source
import main

def Main():
    pathInput = "/home/app/file/input/1_jp.jpg"
    pathOutput = "/home/app/file/output/preprocessor/1_jp.jpg"

    image = main.open(pathInput)

    _, _, _, imageResize, _ = main.resize(image, 2048)
    main.write(pathOutput, "_resize", imageResize)

    _, _, _, imageResizeLineHeight, _ = main.resizeLineHeight(image)
    main.write(pathOutput, "_resizeLineHeight", imageResizeLineHeight)

    imageGray = main.gray(imageResizeLineHeight)
    main.write(pathOutput, "_gray", imageGray)

    imageColor = main.color(imageGray)
    main.write(pathOutput, "_color", imageColor)

    imageMedianBlur = main.medianBlur(imageGray)
    main.write(pathOutput, "_medianBlur", imageMedianBlur)

    imageBinarization = main.binarization(imageGray)
    main.write(pathOutput, "_binarization", imageBinarization)

    contourList, hierarchy = main.contour(imageGray)
    print(f"contourList: {contourList} - hierarchy: {hierarchy}")

    imageMask = main.mask(imageGray)
    main.write(pathOutput, "_mask", imageMask)

    imageMaskApply = main.maskApply(imageGray, imageMask)
    main.write(pathOutput, "_maskApply", imageMaskApply)

    imageGamma = main.gamma(imageGray)
    main.write(pathOutput, "_gamma", imageGamma)

    imageNoiseRemove = main.noiseRemove(imageGray)
    main.write(pathOutput, "_noiseRemove", imageNoiseRemove)

    imageBorder = main.addOrRemoveBorder(imageGray)
    main.write(pathOutput, "_border", imageBorder)

    imageHeatmap1 = main.heatmap(main.numpy.random.rand(256, 256))
    imageHeatmap2 = main.heatmap(main.numpy.random.rand(256, 256))
    main.write(pathOutput, "_heatmap1", imageHeatmap1)
    main.write(pathOutput, "_heatmap2", imageHeatmap2)

Main()