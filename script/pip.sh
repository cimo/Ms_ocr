#!/bin/bash

python3 -m pip install pip --upgrade
python3 -m pip install --break-system-packages -r ${PATH_ROOT}src/library/requirement.txt
