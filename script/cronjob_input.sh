#!/bin/bash

path=${PATH_ROOT}file/input/*/

if [ -z "${PATH_ROOT}" ]; then
    exit 1
fi

currentTime=$(date +%s)

for data in ${path}; do
    if [ -f "${data}" ]; then
        statData=$(stat -c %Y "${data}")
        time=$((currentTime - statData))

        if [ "${time}" -gt 600 ]; then
            rm -f "${data}"

            echo "File '${data}' removed."
        fi
    fi
done
