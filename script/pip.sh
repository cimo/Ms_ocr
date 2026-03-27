#!/bin/bash

set -euo pipefail

python3 -m pip install --break-system-packages --upgrade pip
python3 -m pip install --break-system-packages -r ${PATH_ROOT}src/library/requirement.txt
