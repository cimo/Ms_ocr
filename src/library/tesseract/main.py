# Source
import helper

def Main():
    #helper.executeCraft()

    imageGray, imageBox, imageResult = helper.preprocess()

    helper.result(imageGray, imageBox, imageResult)

Main()

# TO DO - Integrate dewarp

#python3 main.py "1_jp.jpg" "jp" "False" "True"
