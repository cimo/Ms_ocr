import os
import cv2
import icu

stderrFileDescriptor = os.dup(2)
nullFileDescriptor = os.open(os.devnull, os.O_WRONLY)

os.dup2(nullFileDescriptor, 2)

import onnxruntime

os.dup2(stderrFileDescriptor, 2)

os.close(nullFileDescriptor)
os.close(stderrFileDescriptor)

def onnxSessionBuild(pathModel):
    option = onnxruntime.SessionOptions()

    option.log_severity_level = 3

    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    option.intra_op_num_threads = max(1, os.cpu_count())
    option.inter_op_num_threads = 1

    option.enable_cpu_mem_arena = True
    option.enable_mem_pattern = True
    option.enable_mem_reuse = True

    option.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

    providerPreferredList = [
        "CUDAExecutionProvider",
        "OpenVINOExecutionProvider",
        "CPUExecutionProvider"
    ]

    providerAvailableList = onnxruntime.get_available_providers()

    providerList = [provider for provider in providerPreferredList if provider in providerAvailableList]

    inference = onnxruntime.InferenceSession(pathModel, sess_options=option, providers=providerList)

    print(f"Provider available: {providerAvailableList}")
    print(f"Provider active: {inference.get_providers()}\n")

    return inference

# Custom
widthWideList = [icu.Char.getPropertyValueEnum(icu.UProperty.EAST_ASIAN_WIDTH, "W"), icu.Char.getPropertyValueEnum(icu.UProperty.EAST_ASIAN_WIDTH, "F")]
lineBreakComplex = icu.Char.getPropertyValueEnum(icu.UProperty.LINE_BREAK, "SA")

def wideCheck(character):
    if character == "":
        return False

    return icu.Char.getIntPropertyValue(character, icu.UProperty.EAST_ASIAN_WIDTH) in widthWideList

def spacelessCheck(character):
    if wideCheck(character):
        return True

    return icu.Char.getIntPropertyValue(character, icu.UProperty.LINE_BREAK) == lineBreakComplex

def whitespaceCheck(character):
    if character == "":
        return False

    return icu.Char.isUWhiteSpace(character)

def sentenceEndCheck(text, levelReferenceLength):
    textClean = sentenceTailStrip(text, levelReferenceLength)

    return len(textClean) > 0 and icu.Char.hasBinaryProperty(textClean[-1:], icu.UProperty.S_TERM)

def sentenceTailStrip(text, levelReferenceLength):
    result = text.strip()

    while len(result) > 0:
        character = result[-1:]

        if icu.Char.charType(character) == icu.UCharCategory.END_PUNCTUATION:
            indexOpen = groupOpenIndex(result, levelReferenceLength)

            result = result[0:indexOpen] if indexOpen >= 0 else result[0:-1]

            continue

        if icu.Char.charType(character) == icu.UCharCategory.FINAL_PUNCTUATION or icu.Char.hasBinaryProperty(character, icu.UProperty.QUOTATION_MARK) or icu.Char.isUWhiteSpace(character):
            result = result[0:-1]

            continue

        break

    return result

def groupOpenIndex(text, levelReferenceLength):
    for a in range(len(text) - 2, len(text) - 2 - levelReferenceLength, -1):
        if a < 0:
            break

        character = text[a]

        if icu.Char.charType(character) == icu.UCharCategory.START_PUNCTUATION:
            return a

        if icu.Char.charType(character) == icu.UCharCategory.END_PUNCTUATION or icu.Char.hasBinaryProperty(character, icu.UProperty.S_TERM):
            break

    return -1

def centerPointCalculate(bboxList):
    return {
        "x": int(round((bboxList[0] + bboxList[2]) / 2)),
        "y": int(round((bboxList[1] + bboxList[3]) / 2))
    }

def boxIntersection(bboxList, bboxOtherList):
    x1 = max(bboxList[0], bboxOtherList[0])
    y1 = max(bboxList[1], bboxOtherList[1])
    x2 = min(bboxList[2], bboxOtherList[2])
    y2 = min(bboxList[3], bboxOtherList[3])

    if x2 <= x1 or y2 <= y1:
        return 0

    return (x2 - x1) * (y2 - y1)

def boxArea(bboxList):
    return (bboxList[2] - bboxList[0]) * (bboxList[3] - bboxList[1])

def boxContainedRemove(boxList, nameKey, levelContained):
    resultList = []

    for a in range(len(boxList)):
        area = boxArea(boxList[a][nameKey])

        isContained = False

        for b in range(len(boxList)):
            if a == b:
                continue

            if boxArea(boxList[b][nameKey]) <= area:
                continue

            if boxIntersection(boxList[a][nameKey], boxList[b][nameKey]) / float(area) >= levelContained:
                isContained = True

                break

        if isContained == False:
            resultList.append(boxList[a])

    return resultList

def imageInkBuild(image):
    return cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

def boxDebugWrite(image, bboxList, pathFile):
    imageDebug = image.copy()

    for a in range(len(bboxList)):
        cv2.rectangle(imageDebug, (bboxList[a][0], bboxList[a][1]), (bboxList[a][2], bboxList[a][3]), (0, 200, 0), 1)

    cv2.imwrite(pathFile, imageDebug)
# Custom
