import sys

# Source
from main import engineActive

language = sys.argv[1]
fileName = sys.argv[2]
uniqueId = sys.argv[3]

engineActive._execute(language, fileName, uniqueId)