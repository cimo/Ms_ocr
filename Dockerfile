FROM node:18.13.0

ARG VERSION_TAG
ARG ENV_NAME
ARG DOMAIN

ENV VERSION_TAG=${VERSION_TAG}
ENV ENV_NAME=${ENV_NAME}
ENV DOMAIN=${DOMAIN}

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NOWARNINGS=yes
ENV TZ="Asia/Tokyo"

COPY ./ /home/root/

RUN cd ~ \
    # No root
    && mkdir -p /home/root/ \
    && chown -R node:node /home/root/ /usr/local/lib/node_modules/ \
    && chmod 775 /home/root/ /usr/local/lib/node_modules/ \
    # Apt
    && apt-get update && apt-get install -y \
    # Clean
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean -y \
    && apt-get autoclean -y \
    && apt-get autoremove -y

USER node

WORKDIR /home/root/

RUN npm install && npm run build

CMD node /home/root/dist/Controller/Server.js

EXPOSE 1980
