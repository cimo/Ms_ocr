import sys

fileName = sys.argv[1]
language = sys.argv[2]
isCuda = sys.argv[3].lower() == "true"
isDebug = sys.argv[4].lower() == "true"
engine = sys.argv[5]
uniqueId = sys.argv[6]

if engine == "paddle":
    from engine_paddle.main import EnginePaddle

    EnginePaddle(fileName, language, isCuda, isDebug, uniqueId)
elif engine == "tesseract":
    from engine_tesseract.main import Main as TesseractMain

    TesseractMain(fileName, language, isCuda, isDebug, uniqueId)