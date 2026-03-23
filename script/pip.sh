#!/bin/bash

set -euo pipefail

python3 -m pip install --break-system-packages --ignore-installed --upgrade pip
python3 -m pip install --break-system-packages --ignore-installed -r ${PATH_ROOT}src/library/requirement.txt
