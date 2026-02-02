#!/bin/bash

export XDG_RUNTIME_DIR="/run/user/$(id -u)"
export DBUS_SESSION_BUS_ADDRESS="unix:path=${XDG_RUNTIME_DIR}/bus"

dbus-daemon --session --address="${DBUS_SESSION_BUS_ADDRESS}" --fork
