import os
import subprocess
import numpy
import cv2
import json

# Source
from preprocessor import main as preprocessor

def _subprocess(fileNameSplit):
    resultLanguage = ""
    resultPsm = 6

    if language == "en":
        resultLanguage = "eng"
    elif language == "jp":
        resultLanguage = "jpn"
    elif language == "jp_vert":
        resultLanguage = "jpn_vert"
        resultPsm = 5

    os.environ["TESSDATA_PREFIX"] = f"{PATH_ROOT}src/library/engine_tesseract/language/"
    
    subprocess.run([
        f"{PATH_ROOT}src/library/engine_tesseract/executable",
        f"{PATH_ROOT}{PATH_FILE_OUTPUT}tesseract/{fileNameSplit}_crop.png",
        f"{PATH_ROOT}{PATH_FILE_OUTPUT}tesseract/{fileNameSplit}",
        f"-l", resultLanguage,
        "--psm", str(resultPsm),
        "--oem", "1",
        "-c", "preserve_interword_spaces=0",
        "-c", "textord_tabfind_vertical_text=0",
        "-c", "textord_min_xheight=10",
        "-c", "classify_enable_learning=1",
        "-c", "classify_enable_adaptive_matcher=1",
        "-c", "classify_use_pre_adapted_templates=0",
        "-c", "classify_bln_numeric_mode=1",
        "-c", "enable_noise_removal=1",
        "-c", "noise_maxperword=16",
        "-c", "noise_maxperblob=16",
        "-c", "tessedit_create_txt=1",
        "-c", "tessedit_create_hocr=0",
        "-c", "tessedit_create_alto=0",
        "-c", "tessedit_create_page_xml=0",
        "-c", "tessedit_create_lstmbox=0",
        "-c", "tessedit_create_tsv=0",
        "-c", "tessedit_create_wordstrbox=0",
        "-c", "tessedit_create_pdf=0",
        "-c", "tessedit_create_boxfile=0"
    ], check=True)

def _jsonCreate(pathJson, left, top, right, bottom, pathText):
    if os.path.exists(pathJson):
        with open(pathJson, "r", encoding="utf-8") as file:
            dataList = json.load(file)
    else:
        dataList = []

    box = {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom
    }
    
    with open(pathText, "r", encoding="utf-8") as file:
        text = file.read().strip()

    dataList.append({
        "box": box,
        "text": text
    })

    with open(pathJson, "w", encoding="utf-8") as file:
        json.dump(dataList, file, ensure_ascii=False, indent=2)

def executeCraft(PATH_ROOT, PATH_FILE_INPUT, PATH_FILE_OUTPUT, fileName, isCuda, isDebug):
    subprocess.run([
        "python3",
        f"{PATH_ROOT}src/library/craft/main.py",
        PATH_ROOT,
        PATH_FILE_INPUT,
        f"{PATH_FILE_OUTPUT}craft/",
        fileName,
        isCuda,
        isDebug
    ], check=True)

def preprocess():
    print(f"Load file: {PATH_ROOT}{PATH_FILE_INPUT}{fileName}\r")

    imageRead = preprocessor.open(f"{PATH_ROOT}{PATH_FILE_INPUT}{fileName}")

    imageGray = preprocessor.gray(imageRead)

    imageBox = imageGray.copy()

    imageResult = imageGray.copy()
    imageResult.fill(255)

    return imageGray, imageBox, imageResult

def result(imageGray, imageBox, imageResult):
    fileNameSplit, _ = os.path.splitext(fileName)

    pathJson = f"{PATH_ROOT}{PATH_FILE_OUTPUT}tesseract/{fileNameSplit}.json"
    pathText = f"{PATH_ROOT}{PATH_FILE_OUTPUT}tesseract/{fileNameSplit}.txt"

    if os.path.exists(pathJson):
        os.remove(pathJson)

    with open(f"{PATH_ROOT}{PATH_FILE_OUTPUT}craft/{fileNameSplit}.txt", "r") as file:
        lineList = file.readlines()

    for _, line in enumerate(lineList):
        coordinateList = list(map(int, line.strip().split(",")))
        boxList = numpy.array(coordinateList).reshape((-1, 2))

        left = int(boxList[0][0])
        top = int(boxList[0][1])
        right = int(boxList[2][0])
        bottom = int(boxList[2][1])

        imageCrop = imageGray[top:bottom, left:right]
        #imageCropFix = _cropFix(imageCrop)

        if isDebug:
            cv2.rectangle(imageBox, (left, top), (right, bottom), (0, 0, 0), 1)

            preprocessor.write(f"{PATH_ROOT}{PATH_FILE_OUTPUT}tesseract/{fileName}", "_box", imageBox)

        imageResult[top:bottom, left:right] = imageCrop

        _, _, _, imageCropResize, _ = preprocessor.resizeLineHeight(imageCrop)

        preprocessor.write(f"{PATH_ROOT}{PATH_FILE_OUTPUT}tesseract/{fileNameSplit}.png", "_crop", imageCropResize)

        _subprocess(fileNameSplit)

        _jsonCreate(pathJson, left, top, right, bottom, pathText)
    
    if os.path.exists(pathText):
        os.remove(pathText)
    
    if isDebug:
        preprocessor.write(f"{PATH_ROOT}{PATH_FILE_OUTPUT}tesseract/{fileName}", "_result", imageResult)
