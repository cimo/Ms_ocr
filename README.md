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
fileName        test
file            "upload field"
language        jp
result          pdf
```

4. Logout

```
url = https://localhost:1045/logout
method = GET
```
