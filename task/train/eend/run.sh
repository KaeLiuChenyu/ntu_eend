#!/usr/bin/env bash

set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="test"

train_config="config/train_eend.yaml"
decode_config="config/decode_eend.yaml"
num_spk=2

./train_scr/train_eend.sh \
    --collar 0.0 \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --ngpu 1 \
    --diar_config "${train_config}" \
    --inference_config "${decode_config}" \
    --inference_nj 5 \
    --local_data_opts "--num_spk ${num_spk}" \
    --num_spk "${num_spk}"\
    "$@"
