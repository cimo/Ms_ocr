FROM node:18.13.0

ARG VERSION_TAG
ARG ENV_NAME
ARG DOMAIN
ARG MS_OCR_SERVER_PORT

ENV VERSION_TAG=${VERSION_TAG}
ENV ENV_NAME=${ENV_NAME}
ENV DOMAIN=${DOMAIN}
ENV MS_OCR_SERVER_PORT=${MS_OCR_SERVER_PORT}

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NOWARNINGS=yes
ENV TZ="Asia/Tokyo"

COPY ./tessdata_best/ /usr/share/tesseract-ocr/4.00/tessdata/
COPY ./ /home/root/

RUN cd ~ \
    # No root
    && mkdir -p /home/root/ \
    && chown -R node:node /home/root/ /usr/local/lib/node_modules/ \
    && chmod 775 /home/root/ /usr/local/lib/node_modules/ \
    # Apt
    && apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    # Clean
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean -y \
    && apt-get autoclean -y \
    && apt-get autoremove -y

USER node

WORKDIR /home/root/

RUN npm install && npm run build

CMD tesseract --version && tesseract --list-langs \
    && node /home/root/dist/Controller/Server.js

EXPOSE ${MS_OCR_SERVER_PORT}
