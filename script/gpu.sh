#!/bin/bash

set -euo pipefail

command -v nvidia-smi > /dev/null && nvidia-smi --query-gpu=name --format=csv,noheader || echo "Host without nvidia GPU support."
