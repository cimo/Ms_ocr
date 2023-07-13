# Ms_ocr

Microservice OCR.

Depend from Ms_cronjob (use the volume "ms_cronjob-volume" for share the certificate).

## Setup WSL

1. Wrinte on terminal:

```
docker compose -f docker-compose.yaml --env-file ./env/local.env up --detach --build --pull "always"
```

2. If you have a proxy execute this command (if you use a certificate put it in "/certificate/proxy/" folder):

```
DOCKERFILE="Dockerfile_local_proxy" docker compose -f docker-compose.yaml --env-file ./env/local.env up --detach --build --pull "always"
```

## Setup DOCKER DESKTOP

1. Wrinte on terminal:

```
docker-compose -f docker-compose.yaml --env-file ./env/local.env up --detach --build --pull "always"
```

2. If you have a proxy execute this command (if you use a certificate put it in "/certificate/proxy/" folder):

```
DOCKERFILE="Dockerfile_local_proxy" docker-compose -f docker-compose.yaml --env-file ./env/local.env up --detach --build --pull "always"
```

## API (Postman)

1. Extract

```
url = https://localhost:1000/msocr/extract

form-data

key             value
---             ---
token_api       1234
file_name       test
file            "upload field"
language        jp
result          pdf
```
