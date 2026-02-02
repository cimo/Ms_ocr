#!/bin/bash

python3 "${PATH_ROOT}src/library/execute.py" "${1}" "${2}" "${3}" "${4}" "${5}" 2>&1 | tee -a "${PATH_ROOT}${MS_O_PATH_LOG}debug.log"
