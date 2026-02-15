#!/bin/bash

p1=$(printf '%s' "${1}" | xargs)
p2=$(printf '%s' "${2}" | xargs)
p3=$(printf '%s' "${3}" | xargs)

if [ -z "${p1}" ] || [ -z "${p2}" ] || [ -z "${p3}" ]
then
    echo "container_execute.sh - Missing parameter."

    exit 1
fi

parameter1="${1}"
parameter2="${2}"
parameter3="${3}"

echo "Copying from volume..."

docker run --rm \
-v cimo_${parameter1}_ms_cronjob-volume:/home/source/:ro \
-v $(pwd)/certificate/:/home/target/ \
alpine sh -c "cp -r /home/source/* /home/target/"

echo "Execute container."

if [ "${parameter2}" = "build-up" ]
then
    if [ "${parameter3}" = "cpu" ]
    then
        docker compose -f docker-compose-cpu.yaml --env-file ./env/${parameter1}.env build --no-cache &&
        docker compose -f docker-compose-cpu.yaml --env-file ./env/${parameter1}.env up --detach --pull always
    elif [ "${parameter3}" = "gpu" ]
    then
        docker compose -f docker-compose-gpu.yaml --env-file ./env/${parameter1}.env build --no-cache &&
        docker compose -f docker-compose-gpu.yaml --env-file ./env/${parameter1}.env up --detach --pull always
    fi
elif [ "${parameter2}" = "up" ]
then
    if [ "${parameter3}" = "cpu" ]
    then
        docker compose -f docker-compose-cpu.yaml --env-file ./env/${parameter1}.env up --detach --pull always
    elif [ "${parameter3}" = "gpu" ]
    then
        docker compose -f docker-compose-gpu.yaml --env-file ./env/${parameter1}.env up --detach --pull always
    fi
fi
