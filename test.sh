#!/bin/bash

[ "$ELV_MODEL_TEST_GPU_TO_USE" != "" ] || ELV_MODEL_TEST_GPU_TO_USE=0

set -x

rm -rf test_output/
mkdir test_output

mkdir -p .cache

podman run --rm --volume=$(pwd)/test:/elv/test:ro --volume=$(pwd)/test_output:/elv/tags --volume=$(pwd)/.cache:/root/.cache --network host --device nvidia.com/gpu=$ELV_MODEL_TEST_GPU_TO_USE ocr test/test.jpg

ex=$?

cd test_output

exit $ex
