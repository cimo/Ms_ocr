#!/bin/bash

XDG_RUNTIME_DIR="/run/user/$(id -u)" \
DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$(id -u)/bus" \
dbus-daemon --session --address="unix:path=/run/user/$(id -u)/bus" --fork