import subprocess

# Source
import helper

'''
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
'''

def Main():
    subprocess.run([
        "python3",
        "../craft/main.py",
        helper.PATH_ROOT,
        helper.PATH_FILE_INPUT,
        helper.PATH_FILE_OUTPUT,
        helper.fileName,
        helper.isCuda,
        helper.isDebug
    ], check=True)

    imageGray, imageRectangle, imageResult, ratio = helper.preprocess()

    coordinateList = helper.readBoxCoordinatesFromFile()

    helper.result(coordinateList, ratio, imageGray, imageRectangle, imageResult)

    helper.tesseract()

Main()

# TO DO - Integrate dewarp

#python3 main.py "test_1.jpg" "en" "" False True