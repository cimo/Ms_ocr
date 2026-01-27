import os
import logging
import ast

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

PATH_ROOT = _checkEnvVariable("PATH_ROOT")
ENGINE = _checkEnvVariable("MS_O_ENGINE")

engineActive = None

if ENGINE == "tesseract":
    from engine_tesseract.main import EngineTesseract

    engineActive = EngineTesseract()
elif ENGINE == "realtime":
    from engine_realtime.main import EngineRealtime

    engineActive = EngineRealtime()
elif ENGINE == "paddle":
    from engine_paddle.main import EnginePaddle

    engineActive = EnginePaddle()