#!/usr/bin/env bash

set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"

train_config="config/train_eend.yaml"
num_spk=2

./train_scr/train_eend.sh \
    --collar 0.0 \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --ngpu 1 \
    --diar_config "${train_config}" \
    --local_data_opts "--num_spk ${num_spk}" \
    --num_spk "${num_spk}"\
    "$@"
