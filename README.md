# Ms_ocr

Microservice OCR.

Depend from Ms_cronjob (use the volume "ms_cronjob-volume" for share the certificate).

## Installation

1. For full build write on terminal:

```
docker compose -f docker-compose.yaml --env-file ./env/local.env build --no-cache \
&& docker compose -f docker-compose.yaml --env-file ./env/local.env up --detach --pull "always"
```

2. For light build (just env variable change) remove the container and write on terminal:

```
docker compose -f docker-compose.yaml --env-file ./env/local.env up --detach --pull "always"
```

## Reset

1. Remove this from the root:

    - .cache
    - .config
    - .local
    - .ms_cronjob-volume
    - .npm
    - .pki
    - node_modules
    - package-lock.json
    - certificate/tls.crt
    - certificate/tls.key
    - certificate/tls.pem

2. Follow the "Installation" instructions.

3. For compile "tesseract" from source write on terminal:
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
language        en
file            "upload field"
isCuda          false
isDebug         true
engine          tesseract
```

4. Download

```
url = https://localhost:1045/download
method = POST

json

key             value
---             ---
"engine":       "tesseract",
"uniqueId":     "mgsr5rff-2sp0c9",
"pathFile":     "export/1_jp.pdf"
```

5. Logout

```
url = https://localhost:1045/logout
method = GET
```
