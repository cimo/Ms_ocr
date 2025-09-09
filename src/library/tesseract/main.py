# Source
import helper

def Main():
    helper.executeCraft()

    imageGray, imageRectangle, imageResult, ratio = helper.preprocess()

    coordinateList = helper.readBoxCoordinatesFromFile()

    helper.result(coordinateList, ratio, imageGray, imageRectangle, imageResult)

    helper.execute()

Main()

# TO DO - Integrate dewarp
#python3 main.py "test_1.jpg" "en" False True