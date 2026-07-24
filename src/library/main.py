import sys
sys.dont_write_bytecode = True

import os
import logging
import ast

# Soruce
from onnx.document_scanner.server import Server

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

server = Server()
server.execute(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])