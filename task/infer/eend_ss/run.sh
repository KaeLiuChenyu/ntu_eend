#!/usr/bin/env bash

train_config="/export/home2/tuantd/diarization/ntu_eend/task/train/eend_ss/config/train_eend_ss.yaml"
model_file="/export/home2/tuantd/diarization/ntu_eend/task/infer/eend_ss/checkpoints/valid.si_snr_loss.best_old.pth"
threshold=0.7

generate_rttm=true

python ./infer_scr/eend_ss_inference.py \
      --ngpu 0 \
      --fs 8k \
      --data_path_and_name_and_type "data/test/wav.scp,speech_mix,sound" \
      --train_config $train_config \
      --model_file $model_file \
      --output_dir "result" \
      --config "config/eend_ss_inference.yaml"

if [[ ${generate_rttm} == "true" ]]; then
python ./utils/make_rttm.py \
      --median 11 \
      --threshold $threshold \
      --frame_shift 128 \
      --subsampling 1 \
      "/result/diarize.scp" \
      "/result/pre.rttm"
