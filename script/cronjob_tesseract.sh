#!/bin/bash

pathOutput="${PATH_ROOT}${MS_O_PATH_FILE}output/tesseract/"

if [ -z "${PATH_ROOT}" ]
then
    exit 1
fi

currentTime=$(date +%s)

for data in "${pathOutput}"*/
do
    if [ -d "${data}" ]
    then
        statData=$(stat -c %Y "${data}")
        time=$((${currentTime} - ${statData}))

        if [ ${time} -gt 600 ]
        then
            rm -rf "${data}"

            echo "Folder '${data}' removed."
        fi
    fi
done
