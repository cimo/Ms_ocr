#!/bin/bash

. "${PATH_ROOT}.venv/bin/activate"

p1=$(printf '%s' "${1}" | xargs)
p2=$(printf '%s' "${2}" | xargs)
p3=$(printf '%s' "${3}" | xargs)
p4=$(printf '%s' "${4}" | xargs)

if [ "$#" -lt 5 ]
then
    echo -e "\n❌ command1.sh - Missing parameter."

    exit 1
fi

parameter1="${1}"
parameter2="${2}"
parameter3="${3}"
parameter4="${4}"

python3 "${PATH_ROOT}src/library/main.py" "${parameter1}" "${parameter2}" "${parameter3}" "${parameter4}" 2>&1 | tee -a "${PATH_ROOT}${MS_O_PATH_LOG}debug.log"
