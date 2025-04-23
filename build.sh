#!/bin/bash

if ! ssh-add -l ; then
    echo ssh agent does not have any identities loaded, will not be able to build
    echo add them by running ssh-add on your local machine, or on the remote if you have keys there
    echo you may also need to restart vs code and the remote server for this to work
    exit 1
fi



SCRIPT_PATH="$(dirname "$(realpath "$0")")"
MODEL_PATH=$(yq -r .storage.model_path $SCRIPT_PATH/config.yml)

mkdir -p $SCRIPT_PATH/models
rsync --progress --update --times --recursive --links --delete $MODEL_PATH $SCRIPT_PATH/models/

podman build --format docker -t ocr . --network host --build-arg SSH_AUTH_SOCK=/tmp/ssh-auth-sock --volume "${SSH_AUTH_SOCK}:/tmp/ssh-auth-sock"
