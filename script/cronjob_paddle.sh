#!/bin/bash

path=${PATH_ROOT}file/output/paddle/*/
currentTime=$(date +%s)

for data in ${path}; do
    if [ -d "${data}" ]; then
        statData=$(stat -c %Y "${data}")
        time=$((currentTime - statData))

        if [ "${time}" -gt 600 ]; then
            echo "${data}: old."
        fi
    fi
done
