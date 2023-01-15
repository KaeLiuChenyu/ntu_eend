#!/usr/bin/env bash

set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"

train_config="config/train_eend_ss.yaml"
train_config2="config/adapt_train_eend_ss.yaml"
num_spk=2

pretrain_model="exp/diar_train_diar_eda_5_raw_max_epoch250/valid.acc.ave_10best.pth"

pretrain_stage=true
adapt_stage=false

if [[ ${pretrain_stage} == "true" ]]; then
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
fi

# Modify "--diar_args "--init_param <path of the pre-trained model>""
if [[ ${adapt_stage} == "true" ]]; then
./train_scr/train_eend_ss.sh \
    --use_noise_ref true \
    --collar 0.0 \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --ngpu 1 \
    --diar_config "${train_config2}" \
    --audio_format wav \
    --local_data_opts "--num_spk ${num_spk}" \
    --spk_num 3\
    --hop_length 64 \
    --frame_shift 64 \
    --stage 1 \
    --stop_stage 10 \
    --diar_args "--init_param ${pretrain_model}" \
    "$@"
fi
