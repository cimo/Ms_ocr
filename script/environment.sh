pathEnvFile="./env/${ENV_NAME}.env"

if [ -f "${pathEnvFile}" ]
then
    while IFS= read -r line
    do
        if [ -n "${line}" ]
        then
            key=$(echo "${line}" | cut -d "=" -f 1)

            value=$(echo "${line}" | cut -d "=" -f 2-)
            value=$(echo "${value}" | sed "s/^'\(.*\)'$/\1/")

            if [ -z "$(echo ${!key})" ]
            then
                echo "File env export: ${key}=${value}"

                export "${key}=${value}"
            fi
        fi
    done < "${pathEnvFile}"
fi

export PATH="${PATH}:${PATH_ROOT}.local/bin/:/root/.local/bin/"
