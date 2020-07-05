#!/usr/bin/env bash

set -xe

##################################################
# Install llvm-config on Ubuntu

sudo apt-get update
sudo apt-get install -y llvm-dev

##################################################
# Install onnxruntime

ONNXRUNTIME_VERSION=${ONNXRUNTIME_VERSION:-1.2.0}

ONNXRUNTIME_RELEASE="onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}"
ONNXRUNTIME_RELEASE_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ONNXRUNTIME_RELEASE}.tgz"

curl -L "$ONNXRUNTIME_RELEASE_URL" --output "$ONNXRUNTIME_RELEASE.tgz"

tar -xzf "$ONNXRUNTIME_RELEASE.tgz" "$ONNXRUNTIME_RELEASE"

export ONNXRUNTIME_LIB_DIR="$PWD/$ONNXRUNTIME_RELEASE/lib/"
export ONNXRUNTIME_INCLUDE_DIR="$PWD/$ONNXRUNTIME_RELEASE/include"
