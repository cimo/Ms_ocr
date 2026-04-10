# Ms_ocr
Microservice OCR.

Depend on "Ms_cronjob" (use "ms_cronjob-volume" to share the certificate).

It's possible to use a custom certificate instead of self‑signed.
Just add it to the "/certificate/custom/" folder and change the env variable before build the container.

## Info:
- Cross platform (Windows, Linux)
- WSLg for WSL2 (Run linux GUI app directly in windows) with full nvidia GPU host support.
- Paddle (default), Tesseract.

## Installation
1. For build and up with GPU write on terminal:
```
bash docker/container_execute.sh "local" "build-up" "gpu"
```

2. For build and up with CPU write on terminal:
```
bash docker/container_execute.sh "local" "build-up" "cpu"
```

3. Just for up write on terminal:
```
bash docker/container_execute.sh "local" "up" "xxx"
```

## GPU
1. When the container start, a message appears that indicates the GPU status:

    NVIDIA GeForce RTX 3060 - (Host GPU available)

## Reset
1. Delete this from the root:
    - .cache
    - .config
    - .cuda
    - .local
    - .npm
    - .nv
    - .paddlex
    - .pki
    - .venv
    - dist
    - node_modules
    - package-lock.json

2. Follow the "Installation" instructions.

## API (Postman)
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
language        "" (Only for "engine_tesseract" need be populated.)
file            jp_1.jpg
searchText      "" (Only for "engine_realtime" need be populated.)
mode            "file" (Or "data".)
```

4. Download
```
url = https://localhost:1045/api/download
method = POST

json

key             value
---             ---
"uniqueId":     "1234",
"pathFile":     "export/jp_1_result.pdf"
```

5. Logout
```
url = https://localhost:1045/logout
method = GET
```
