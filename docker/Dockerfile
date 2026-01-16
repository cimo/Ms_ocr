FROM ubuntu:22.04

ARG _ENV_NAME
ARG _DOMAIN
ARG _TIME_ZONE
ARG _LANG
ARG _SERVER_PORT
ARG _PATH_ROOT
ARG _VERSION_TAG

ENV ENV_NAME=${_ENV_NAME}
ENV DOMAIN=${_DOMAIN}
ENV TZ=${_TIME_ZONE}
ENV LANG=${_LANG}
ENV SERVER_PORT=${_SERVER_PORT}
ENV PATH_ROOT=${_PATH_ROOT}
ENV VERSION_TAG=${_VERSION_TAG}

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NOWARNINGS=yes
ENV SUDO_FORCE_REMOVE=yes
ENV NODE_OPTIONS=--use-openssl-ca
ENV PATH="${PATH}:${PATH_ROOT}.local/bin/:/root/.local/bin/"
ENV PYTHONPATH="${PATH_ROOT}"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json

COPY ./.ms_cronjob-volume/certificate/* /usr/local/share/ca-certificates/
COPY ./docker/NotoSansJP-Regular.ttf /usr/share/fonts/truetype/noto/
COPY ./ ${PATH_ROOT}

RUN cd ~ \
    # No root
    && useradd -U node -d ${PATH_ROOT} -s /bin/bash \
    && mkdir -p ${PATH_ROOT} \
    && chown -R node:node ${PATH_ROOT} \
    && chmod -R 775 ${PATH_ROOT} \
    # Update
    && VERSION_CODENAME=$(grep VERSION_CODENAME= /etc/os-release | cut -d= -f2) \
    && chmod -R 0644 /usr/local/share/ca-certificates/ \
    && apt-get update && apt-get install -y ca-certificates tzdata curl media-types \
    && update-ca-certificates \
    && ln -sf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone \
    && echo "application/typescript ts" >> /etc/mime.types \
    # NodeJs
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get update && apt-get install -y \
    nodejs \
    # Python
    && apt-get update && apt-get install -y \
    python3-pip \
    # Dbus
    && apt-get update && apt-get install -y \
    dbus \
    dbus-x11 \
    # GPU
    && apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgtk-3-0 \
    libnss3 \
    libnspr4 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxrandr2 \
    libfreetype6 \
    libfontconfig1 \
    libasound2 \
    libgomp1 \
    libgbm1 \
    libglu1-mesa \
    libvulkan1 \
    mesa-vulkan-drivers \
    mesa-utils \
    xdg-utils \
    libxcb1 \
    libx11-xcb1 \
    libcairo2 \
    # Tesseract
    && apt-get update && apt-get install -y \
    cmake \
    g++ \
    pkg-config \
    libleptonica-dev \
    libtiff-dev \
    libgl1 \
    ccache \
    # Clean
    && apt-get remove --purge -y sudo \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean -y \
    && apt-get autoclean -y \
    && apt-get autoremove -y

USER node

WORKDIR ${PATH_ROOT}

# Pip
RUN python3 -m pip install pip --upgrade \
    && python3 -m pip install --user --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org -r ${PATH_ROOT}src/library/requirements.txt \
    # Npm
    && rm -rf ${PATH_ROOT}node_modules/ ${PATH_ROOT}package-lock.json \
    && npm install \
    && npm run build

# Command
CMD command -v nvidia-smi > /dev/null && nvidia-smi --query-gpu=name --format=csv,noheader || echo "Host without nvidia GPU." \
    && command -v vulkaninfo > /dev/null && vulkaninfo | grep "deviceName" || echo "Host without vulkan support." \
    && command -v glxinfo > /dev/null && glxinfo | grep "OpenGL renderer string" || echo "Host without opengl support." \
    # Engine package
    && bash ${PATH_ROOT}script/engine_package.sh \
    # Library
    && python3 ${PATH_ROOT}src/library/main.py \
    # Node
    && node ${PATH_ROOT}dist/src/controller/Server.js

EXPOSE ${SERVER_PORT}
