#!/usr/bin/env bash

set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"

train_config1="config/train_eend_eda.yaml"
train_config2="config/adapt_train_eend_eda.yaml"

pretrain_model="exp/diar_train_diar_eda_5_raw_max_epoch250/valid.acc.ave_10best.pth"

pretrain_stage=true
adapt_stage=false
# If you want to run only one of the stages (e.g., the adaptation stage),
# set "false" to the one you don't want to run (e.g., the pre-training stage)

if [[ ${pretrain_stage} == "true" ]]; then
./train_scr/train_eend_eda.sh \
    --collar 0.0 \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --ngpu 1 \
    --diar_config "${train_config1}" \
    --local_data_opts "--num_spk 2" \
    --stop_stage 5 \
    "$@"
fi

# Modify "--diar_args "--init_param <path of the pre-trained model>""
# according to the actual path of your experiment.
if [[ ${adapt_stage} == "true" ]]; then
./train_scr/train_eend_eda.sh \
    --collar 0.0 \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --ngpu 1 \
    --diar_config "${train_config2}" \
    --local_data_opts "--stage 2" \
    --diar_args "--init_param ${pretrain_model}" \
    --diar_tag "train_diar_eda_adapt_raw" \
    --num_spk "3"\
    "$@"
fi
