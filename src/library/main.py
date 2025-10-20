import sys

language = sys.argv[1]
fileName = sys.argv[2]
isCuda = sys.argv[3].lower() == "true"
isDebug = sys.argv[4].lower() == "true"
engine = sys.argv[5]
uniqueId = sys.argv[6]

if engine == "paddle":
    from engine_paddle.main import EnginePaddle

    EnginePaddle(language, fileName, isCuda, isDebug, uniqueId)
elif engine == "tesseract":
    from engine_tesseract.main import EngineTesseract

    EngineTesseract(language, fileName, isCuda, isDebug, uniqueId)