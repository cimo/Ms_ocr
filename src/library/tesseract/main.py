# Source
import helper

def Main():
    helper.executeCraft()

    imageGray, imageRectangle, imageResult, ratio = helper.preprocess()

    coordinateList = helper.readBoxCoordinateFromFile()

    helper.result(coordinateList, ratio, imageGray, imageRectangle, imageResult)

    helper.execute()

Main()

# TO DO - Integrate dewarp