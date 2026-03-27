#!/bin/bash

set -euo pipefail

path="${PATH_ROOT}${MS_O_PATH_FILE}output/engine_realtime/"

currentTime=$(date +%s)

for data in "${path}"*
do
    if [ -e "${data}" ]
    then
        statData=$(stat -c %Y "${data}")
        time=$((${currentTime} - ${statData}))

        if [ ${time} -gt "${MS_O_PERSISTENCE_SECOND}" ]
        then
            if [ -d "${data}" ]
            then
                rm -rf "${data}"

                echo -e "\nFolder '${data}' deleted."
            else
                rm -f "${data}"
                
                echo -e "\nFile '${data}' deleted."
            fi
        fi
    fi
done