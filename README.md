# Ms_ocr

Microservice OCR.

Depend on "Ms_cronjob" (use "ms_cronjob-volume" to share the certificate).

It's possible to use a custom certificate instead of "Ms_cronjob", just add it to the "certificate" folder before build the container.

## Info:

-   Cross platform (Windows, Linux)
-   WSLg for WSL2 (Run linux GUI app directly in windows) with full nvidia GPU host support.
-   Tesseract, Paddle, Realtime (default).

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

1. Remove this from the root:
    - .cache
    - .config
    - .cuda
    - .local
    - .ms_cronjob-volume/certificate
    - .npm
    - .nv
    - .paddlex
    - .pki
    - dist
    - node_modules
    - package-lock.json

2. Follow the "Installation" instructions.

## Tesseract

1. For compile "tesseract" from source with custom setting write on terminal (standard version is already deployed):
    
    ```
    cd src/library/engine_tesseract/
    tar -xvzf 5.5.1.tar.gz
    cd tesseract-5.5.1/
    mkdir build && cd build
    cmake .. -DENABLE_NATIVE=OFF -DBUILD_TRAINING_TOOLS=OFF -DHAVE_LIBCURL=OFF -DGRAPHICS_DISABLED=OFF -DBUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    ```

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
dataType        "file" (Or "data".)
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
