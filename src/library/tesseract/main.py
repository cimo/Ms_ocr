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
    helper.executeCraft()

    imageGray, imageRectangle, imageResult, ratio = helper.preprocess()

    coordinateList = helper.readBoxCoordinatesFromFile()

    helper.result(coordinateList, ratio, imageGray, imageRectangle, imageResult)

    helper.execute()

Main()

# TO DO - Integrate dewarp

#python3 main.py "test_1.jpg" "en" "pdf" False True

#export TESSDATA_PREFIX=/home/app/src/library/tesseract/language/
#./executable /home/app/file/output/tesseract/test_1_result.jpg  /home/app/file/output/tesseract/test_1 -l eng --oem 1 --psm 6 -c preserve_interword_spaces=1 -c page_separator='' -c tessedit_char_blacklist='〇' pdf