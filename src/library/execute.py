import sys

# Source
from main import engineActive

language = sys.argv[1]
fileName = sys.argv[2]
isDebug = sys.argv[3].lower() == "true"
uniqueId = sys.argv[4]

engineActive._execute(language, fileName, isDebug, uniqueId)