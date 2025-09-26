import os
from paddleocr import LayoutDetection

os.makedirs("/home/app/file/output/paddle/", exist_ok=True)

pathModelBlockLayout = "/home/app/src/library/paddle/PP-DocLayout_plus-L/"
pathImage = "/home/app/file/input/1_jp.jpg"

modelBlockLayout = LayoutDetection(model_dir=pathModelBlockLayout, model_name="PP-DocLayout_plus-L")
outputBlockLayoutList = modelBlockLayout.predict(pathImage, batch_size=1)

for outputBlockLayout in outputBlockLayoutList:
    outputBlockLayout.save_to_img(save_path="/home/app/file/output/paddle/layout.jpg")
    outputBlockLayout.save_to_json(save_path="/home/app/file/output/paddle/layout.json")