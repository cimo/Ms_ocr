import os
import logging
import ast
import sys

def _checkEnvVariable(varKey):
    if os.environ.get(varKey) is None:
        logging.exception("Environment variable %s do not exist", varKey)
    else:
        if os.environ.get(varKey).lower() == "true":
            return True
        if os.environ.get(varKey).lower() == "false":
            return False
        if os.environ.get(varKey).isnumeric():
            return int(os.environ.get(varKey))
        if os.environ.get(varKey).startswith("[") and os.environ.get(varKey).endswith("]"):
            return ast.literal_eval(os.environ.get(varKey))

    return os.environ.get(varKey)

ENGINE = _checkEnvVariable("MS_O_ENGINE")

language = sys.argv[1]
fileName = sys.argv[2]
isCuda = sys.argv[3].lower() == "true"
isDebug = sys.argv[4].lower() == "true"
uniqueId = sys.argv[5]

if ENGINE == "tesseract":
    from engine_tesseract.main import EngineTesseract

    EngineTesseract(language, fileName, isCuda, isDebug, uniqueId)
elif ENGINE == "paddle":
    from engine_paddle.main import EnginePaddle

    EnginePaddle(language, fileName, isCuda, isDebug, uniqueId)