# Ms_ocr

Microservice OCR.

Depend from Ms_cronjob (use the volume "ms_cronjob-volume" for share the certificate).

## Info:

-   Cross platform (Windows, Linux)
-   X11 for WSL2 (Run linux GUI app directly in windows) with full GPU host support.
-   Craft, Tesseract, Paddle (default).

## Installation

1. For full build with GPU write on terminal:

```
docker compose -f docker-compose-gpu.yaml --env-file ./env/local.env build --no-cache \
&& docker compose -f docker-compose-gpu.yaml --env-file ./env/local.env up --detach --pull "always"
```

2. For full build with CPU write on terminal:

```
docker compose -f docker-compose-cpu.yaml --env-file ./env/local.env build --no-cache \
&& docker compose -f docker-compose-cpu.yaml --env-file ./env/local.env up --detach --pull "always"
```

3. For light build (just env variable change) remove the container and write on terminal:

```
docker compose -f docker-compose-xxx.yaml --env-file ./env/local.env up --detach --pull "always"
```

## GPU

1. When the container startup are present 3 message that indicate your GPU status:

Host without nvidia GPU. - (No Nvidia driver/GPU available)
Host without vulkan support. - (No library vulkan available for your GPU)
OpenGL renderer string: llvmpipe (LLVM 15.0.7, 256 bits) - (OpenGL emulate on CPU)

## Reset

1. Remove this from the root:

    - .cache
    - .local
    - .ms_cronjob-volume
    - .npm
    - .paddlex
    - node_modules
    - package-lock.json
    - certificate/tls.crt
    - certificate/tls.key
    - certificate/tls.pem

2. Follow the "Installation" instructions.

## Tesseract

1. For compile "tesseract" from source with custom setting write on terminal:
    
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

3. Extract

```
url = https://localhost:1045/extract
method = POST

form-data

key             value
---             ---
language        (For paddle is empty. For tesseract will be like "en", "jp" or "jp_vert")
file            jp_1.jpg
```

4. Download

```
url = https://localhost:1045/download
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
