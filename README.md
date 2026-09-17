# Ms_file_data_extractor
Microservice file data extractor.

Depend on "Ms_cronjob" (use "ms_cronjob-volume" to share the certificate).

It's possible to use a custom certificate instead of self‑signed.
Just add it to the "/certificate/custom/" folder and change the env variable before build the container.

## Info:
- Cross platform (Windows, Linux)
- WSLg for WSL2 (Run linux GUI app directly in windows) with full nvidia GPU host support.
- Onnx: (PP-DocLayout_plus-L, PP-OCRv6_medium_det, PP-OCRv6_medium_rec).

## Installation
1. For build and up with GPU write on host terminal:
```
bash docker/container_execute.sh "local" "build-up" "gpu"
```

2. For build and up with CPU write on host terminal:
```
bash docker/container_execute.sh "local" "build-up" "cpu"
```

3. Just for up with GPU write on host terminal:
```
bash docker/container_execute.sh "local" "up" "gpu"
```

4. Just for up with CPU write on host terminal:
```
bash docker/container_execute.sh "local" "up" "cpu"
```

## Reset
1. Delete this from the root:
    - .cache
    - .cuda
    - .local
    - .npm
    - .nv
    - .pki
    - .venv
    - dist
    - node_modules
    - package-lock.json

2. Follow the "Installation" instructions.

## Api
1. Info
```
url = https://localhost:1045/info
method = GET
```

2. Login
```
url = https://localhost:1045/login
method = GET
```

3. Extract data
```
url = https://localhost:1045/api/extract
method = POST

form-data

key             value
---             ---
file            jp_1.jpg
password        "" (only on pdf with password)
searchText      ""
```

4. Logout
```
url = https://localhost:1045/logout
method = GET
```
