#!/usr/bin/env bash

set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="test"


./prepare_librimix.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --nj 1 \
    "$@"