#!/bin/bash


set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}
SECONDS=0

# General configuration
stage=1              # Processes starts from the specified stage.
stop_stage=10000     # Processes is stopped at the specified stage.
skip_data_prep=false # Skip data preparation stages
skip_train=false     # Skip training stages
skip_eval=false      # Skip decoding and evaluation stages
skip_upload=true     # Skip packing and uploading stages
skip_upload_hf=true # Skip uploading to hugging face stages.
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes
nj=32                # The number of parallel jobs.
dumpdir=data         # Directory to dump features.
inference_nj=32      # The number of parallel jobs in decoding.
gpu_inference=false  # Whether to perform gpu decoding.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# Feature extraction related
feats_type=raw       # Feature type (raw or fbank_pitch).
audio_format=flac    # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
fs=8k                # Sampling rate.
hop_length=128       # Hop length in sample number
min_wav_duration=0.1 # Minimum duration in second

# diar model related
diar_tag=    # Suffix to the result dir for diar model training.
diar_config= # Config for diar model training.
diar_args=   # Arguments for diar model training, e.g., "--max_epoch 10".
             # Note that it will overwrite args in diar config.
feats_normalize=global_mvn # Normalizaton layer type.
num_spk=2    # Number of speakers in the input audio

# diar related
inference_config= # Config for diar model inference
inference_model=valid.acc.best.pth
inference_tag=    # Suffix to the inference dir for diar model inference
download_model=   # Download a model from Model Zoo and use it for diarization.

# Upload model related
hf_repo=

# scoring related
collar=0         # collar for der scoring
frame_shift=128  # frame shift to convert frame-level label into real time
                 # this should be aligned with frontend feature extraction

# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
valid_set=       # Name of development set.
test_sets=       # Names of evaluation sets. Multiple items can be specified.
diar_speech_fold_length=800 # fold_length for speech data during diar training
                            # Typically, the label also follow the same fold length
lang=noinfo      # The language type of corpus.


help_message=$(cat << EOF
Usage: $0 --train-set <train_set_name> --valid-set <valid_set_name> --test_sets <test_set_names>

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --skip_train     # Skip training stages (default="${skip_train}").
    --skip_eval      # Skip decoding and evaluation stages (default="${skip_eval}").
    --skip_upload    # Skip packing and uploading stages (default="${skip_upload}").
    --ngpu           # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes      # The number of nodes
    --nj             # The number of parallel jobs (default="${nj}").
    --inference_nj   # The number of parallel jobs in inference (default="${inference_nj}").
    --gpu_inference  # Whether to use gpu for inference (default="${gpu_inference}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --expdir         # Directory to save experiments (default="${expdir}").
    --python         # Specify python to execute espnet commands (default="${python}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Feature extraction related
    --feats_type       # Feature type (only support raw currently).
    --audio_format     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw, default="${audio_format}").
    --fs               # Sampling rate (default="${fs}").
    --hop_length       # Hop length in sample number (default="${hop_length}")
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").


    # Diarization model related
    --diar_tag        # Suffix to the result dir for diarization model training (default="${diar_tag}").
    --diar_config     # Config for diarization model training (default="${diar_config}").
    --diar_args       # Arguments for diarization model training, e.g., "--max_epoch 10" (default="${diar_args}").
                      # Note that it will overwrite args in diar config.
    --feats_normalize # Normalizaton layer type (default="${feats_normalize}").
    --num_spk         # Number of speakers in the input audio (default="${num_spk}")

    # Diarization related
    --inference_config # Config for diar model inference
    --inference_model  # diarization model path for inference (default="${inference_model}").
    --inference_tag    # Suffix to the inference dir for diar model inference
    --download_model   # Download a model from Model Zoo and use it for diarization (default="${download_model}").

    # Scoring related
    --collar      # collar for der scoring
    --frame_shift # frame shift to convert frame-level label into real time

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set               # Name of training set (required).
    --valid_set               # Name of development set (required).
    --test_sets               # Names of evaluation sets (required).
    --diar_speech_fold_length # fold_length for speech data during diarization training  (default="${diar_speech_fold_length}").
EOF
)

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(utils/print_args.py $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

# Check required arguments
[ -z "${train_set}" ] && { log "${help_message}"; log "Error: --train_set is required"; exit 2; };
[ -z "${valid_set}" ] &&   { log "${help_message}"; log "Error: --valid_set is required"  ; exit 2; };

# The directory for dataset
data_feats=${dumpdir}
# The directory used for collect-stats mode
diar_stats_dir="${expdir}/stats"
# The directory used for training commands
diar_exp="${expdir}/train"

workdir=$(cd $(dirname $0); pwd)


# ========================== Main stages start from here. ==========================



if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    _diar_train_dir="${data_feats}/${train_set}"
    _diar_valid_dir="${data_feats}/${valid_set}"
    log "Stage 1: Diarization collect stats: train_set=${_diar_train_dir}, valid_set=${_diar_valid_dir}"

    _opts=
    if [ -n "${diar_config}" ]; then
        # To generate the config file: e.g.
        #   % python3 -m espnet2.bin.diar_train --print_config --optim adam
        _opts+="--config ${diar_config} "
    fi

    _feats_type="$(<${_diar_train_dir}/feats_type)"
    if [ "${_feats_type}" = raw ]; then
        _scp=wav.scp
        if [[ "${audio_format}" == *ark* ]]; then
            _type=kaldi_ark
        else
            # "sound" supports "wav", "flac", etc.
            _type=sound
        fi
        _opts+="--frontend_conf fs=${fs} "
        _opts+="--frontend_conf hop_length=${hop_length} "
    else
        echo "does not support other feats_type (i.e., ${_feats_type}) now"
    fi

    _opts+="--num_spk ${num_spk} "

    # 1. Split the key file
    _logdir="${diar_stats_dir}/logdir"
    mkdir -p "${_logdir}"

    # Get the minimum number among ${nj} and the number lines of input files
    _nj=$(min "${nj}" "$(<${_diar_train_dir}/${_scp} wc -l)" "$(<${_diar_valid_dir}/${_scp} wc -l)")

    key_file="${_diar_train_dir}/${_scp}"
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/train.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    key_file="${_diar_valid_dir}/${_scp}"
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/valid.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    # 2. Generate run.sh
    log "Generate '${diar_stats_dir}/run.sh'. You can resume the process from stage 1 using this script"
    mkdir -p "${diar_stats_dir}"; echo "${run_args} --stage 1 \"\$@\"; exit \$?" > "${diar_stats_dir}/run.sh"; chmod +x "${diar_stats_dir}/run.sh"

    # 3. Submit jobs
    log "Diarization collect-stats started... log: '${_logdir}/stats.*.log'"

    # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
    #       but it's used only for deciding the sample ids.

    # shellcheck disable=SC2086
    utils/run.pl JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
        python $workdir/train_eend.py \
            --collect_stats true \
            --use_preprocessor true \
            --train_data_path_and_name_and_type "${_diar_train_dir}/${_scp},speech,${_type}" \
            --train_data_path_and_name_and_type "${_diar_train_dir}/espnet_rttm,spk_labels,rttm" \
            --valid_data_path_and_name_and_type "${_diar_valid_dir}/${_scp},speech,${_type}" \
            --valid_data_path_and_name_and_type "${_diar_valid_dir}/espnet_rttm,spk_labels,rttm" \
            --train_shape_file "${_logdir}/train.JOB.scp" \
            --valid_shape_file "${_logdir}/valid.JOB.scp" \
            --output_dir "${_logdir}/stats.JOB" \
            ${_opts} ${diar_args} || { cat "${_logdir}"/stats.1.log; exit 1; }

    # 4. Aggregate shape files
    _opts=
    for i in $(seq "${_nj}"); do
        _opts+="--input_dir ${_logdir}/stats.${i} "
    done
    # shellcheck disable=SC2086
    python utils/aggregate_stats_dirs.py ${_opts} --output_dir "${diar_stats_dir}"

fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    _diar_train_dir="${data_feats}/${train_set}"
    _diar_valid_dir="${data_feats}/${valid_set}"
    log "Stage 2: Diarization Training: train_set=${_diar_train_dir}, valid_set=${_diar_valid_dir}"

    _opts=
    if [ -n "${diar_config}" ]; then
        # To generate the config file: e.g.
        #   % python3 -m espnet2.bin.diar_train --print_config --optim adam
        _opts+="--config ${diar_config} "
    fi

    _feats_type="$(<${_diar_train_dir}/feats_type)"
    if [ "${_feats_type}" = raw ]; then
        _scp=wav.scp
        # "sound" supports "wav", "flac", etc.
        if [[ "${audio_format}" == *ark* ]]; then
            _type=kaldi_ark
        else
            _type=sound
        fi
        _fold_length="$((diar_speech_fold_length * 100))"
        _opts+="--frontend_conf fs=${fs} "
        _opts+="--frontend_conf hop_length=${hop_length} "
    else
        echo "does not support other feats_type (i.e., ${_feats_type}) now"
    fi

    if [ "${feats_normalize}" = global_mvn ]; then
        # Default normalization is utterance_mvn and changes to global_mvn
        _opts+="--normalize=global_mvn --normalize_conf stats_file=${diar_stats_dir}/train/feats_stats.npz "
    fi

    _opts+="--num_spk ${num_spk} "

    _opts+="--train_data_path_and_name_and_type ${_diar_train_dir}/${_scp},speech,${_type} "
    _opts+="--train_data_path_and_name_and_type ${_diar_train_dir}/espnet_rttm,spk_labels,rttm "
    _opts+="--train_shape_file ${diar_stats_dir}/train/speech_shape "
    _opts+="--train_shape_file ${diar_stats_dir}/train/spk_labels_shape "

    _opts+="--valid_data_path_and_name_and_type ${_diar_valid_dir}/${_scp},speech,${_type} "
    _opts+="--valid_data_path_and_name_and_type ${_diar_valid_dir}/espnet_rttm,spk_labels,rttm "
    _opts+="--valid_shape_file ${diar_stats_dir}/valid/speech_shape "
    _opts+="--valid_shape_file ${diar_stats_dir}/valid/spk_labels_shape "

    log "Generate '${diar_exp}/run.sh'. You can resume the process from stage 2 using this script"
    mkdir -p "${diar_exp}"; echo "${run_args} --stage 2 \"\$@\"; exit \$?" > "${diar_exp}/run.sh"; chmod +x "${diar_exp}/run.sh"

    # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case
    log "Diarization training started... log: '${diar_exp}/train.log'"

    # shellcheck disable=SC2086
    ${python} -m ntu_eend.scr.task.launch \
        --log "${diar_exp}"/train.log \
        --ngpu "${ngpu}" \
        --num_nodes "${num_nodes}" \
        --init_file_prefix "${diar_exp}"/.dist_init_ \
        --multiprocessing_distributed true -- \
        python $workdir/train_eend.py \
            --use_preprocessor true \
            --resume true \
            --fold_length "${_fold_length}" \
            --fold_length "${diar_speech_fold_length}" \
            --output_dir "${diar_exp}" \
            ${_opts} ${diar_args}

fi

