#!/usr/bin/env bash

set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"

train_config="config/train_eend_ss.yaml"
num_spk=2

./train_scr/train_eend_ss.sh \
    --use_noise_ref true \
    --collar 0.0 \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --ngpu 1 \
    --diar_config "${train_config}" \
    --audio_format wav \
    --local_data_opts "--num_spk ${num_spk}" \
    --spk_num "${num_spk}"\
    --hop_length 64 \
    --frame_shift 64 \
    --stage 1 \
    --stop_stage 10 \
    "$@"
