#!/bin/bash

SCRIPT_PATH="$(dirname "$(realpath "$0")")"
MODEL_PATH=$(yq -r .storage.model_path $SCRIPT_PATH/config.yml)

mkdir -p $SCRIPT_PATH/models
rsync --progress --update --times --recursive --links --delete $MODEL_PATH $SCRIPT_PATH/models/

exec buildscripts/build_container.bash -t "ocr:${IMAGE_TAG:-latest}" . -f Containerfile 