FROM ubuntu:20.04

ARG _ENV_NAME
ARG _DOMAIN
ARG _TIMEZONE
ARG _SERVER_PORT
ARG _VERSION_TAG

ENV ENV_NAME=${_ENV_NAME}
ENV DOMAIN=${_DOMAIN}
ENV TZ=${_TIMEZONE}
ENV VERSION_TAG=${_VERSION_TAG}

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NOWARNINGS=yes

COPY ./docker/nodejs_18_setup.sh /home/nodejs_18_setup.sh
COPY ./docker/tessdata_best/* /usr/share/tesseract-ocr/4.00/tessdata/
COPY ./ /home/root/

RUN cd ~ \
    # NodeJs
    && bash /home/nodejs_18_setup.sh \
    && apt-get update && apt-get install -y nodejs \
    # No root
    && groupadd -g 1000 node \
    && useradd -d /home/root/ -s /bin/bash -u 1000 -g 1000 node \
    && mkdir -p /home/root/ /usr/local/lib/node_modules/ \
    && chown -R node:node /home/root/ /usr/local/lib/node_modules/ /usr/local/bin/ \
    && chmod 775 /home/root/ /usr/local/lib/node_modules/ /usr/local/bin/ \
    # Apt
    && apt-get update && apt-get install -y \
    ca-certificates \
    fonts-noto \
    fonts-noto-cjk \
    libgl1 \
    tesseract-ocr \
    python3-pip \
    # Certificate
    && update-ca-certificates \
    # Clean
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean -y \
    && apt-get autoclean -y \
    && apt-get autoremove -y

USER node

WORKDIR /home/root/

RUN python3 -m pip install pip --upgrade \
    && python3 -m pip install --user --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org -r /home/root/src/library/requirements.txt \
    && npm install && npm run build

CMD tesseract --version \
    && tesseract --list-langs \
    && node /home/root/dist/controller/Server.js

EXPOSE ${_SERVER_PORT}
