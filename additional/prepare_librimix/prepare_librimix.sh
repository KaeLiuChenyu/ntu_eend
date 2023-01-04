#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

train_set=
valid_set=
test_sets=
dumpdir=dump

stage=1
stop_stage=10

spk_num=2
noise_type_num=1
dereverb_ref_num=1

use_dereverb_ref=false
use_noise_ref=false

fs=8k 
audio_format=flac
min_wav_duration=0.1   

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(pyscripts/utils/print_args.py $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi


. ./path.sh
. ./cmd.sh


# Check required arguments
[ -z "${train_set}" ] && { log "${help_message}"; log "Error: --train_set is required"; exit 2; };
[ -z "${valid_set}" ] &&   { log "${help_message}"; log "Error: --valid_set is required"  ; exit 2; };
[ -z "${test_sets}" ] && { log "${help_message}"; log "Error: --test_sets is required"; exit 2; };

data_feats=${dumpdir}/raw



# ========================== Main stages start from here. ==========================


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."
    # [Task dependent] Need to create data.sh for new corpus
    local/data.sh
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then

    log "Stage 2: Format wav.scp: data/ -> ${data_feats}"

    # ====== Recreating "wav.scp" ======
    # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
    # shouldn't be used in training process.
    # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
    # and also it can also change the audio-format and sampling rate.
    # If nothing is need, then format_wav_scp.sh does nothing:
    # i.e. the input file format and rate is same as the output.
    for dset in "${train_set}" "${valid_set}" ${test_sets}; do
        if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
            _suf="/org"
        else
            _suf=""
        fi


        utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}${_suf}/${dset}"
        rm -f ${data_feats}${_suf}/${dset}/{wav.scp,reco2file_and_channel}

        # shellcheck disable=SC2086

        _spk_list=" "
        for i in $(seq ${spk_num}); do
            _spk_list+="spk${i} "
        done

        if $use_noise_ref && [ -n "${_suf}" ]; then
            # references for denoising ("noise1 noise2 ... niose${noise_type_num} ")
            _spk_list+=$(for n in $(seq $noise_type_num); do echo -n "noise$n "; done)
        fi
        if $use_dereverb_ref && [ -n "${_suf}" ]; then
            # references for dereverberation
            _spk_list+=$(for n in $(seq $dereverb_ref_num); do echo -n "dereverb$n "; done)
        fi


        for spk in ${_spk_list} "wav" ; do
            # shellcheck disable=SC2086
            scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                --out-filename "${spk}.scp" \
                --audio-format "${audio_format}" --fs "${fs}" \
                "data/${dset}/${spk}.scp" "${data_feats}${_suf}/${dset}" \
                "${data_feats}${_suf}/${dset}/logs/${spk}" "${data_feats}${_suf}/${dset}/data/${spk}"

        done
        echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"

        # specifics for diarization

        steps/segmentation/convert_utt2spk_and_segments_to_rttm.py \
                "${data_feats}${_suf}/${dset}"/utt2spk \
                "${data_feats}${_suf}/${dset}"/segments \
                "${data_feats}${_suf}/${dset}"/rttm

        # convert standard rttm file into espnet-format rttm (measure with samples)

        pyscripts/utils/convert_rttm.py \
            --rttm "${data_feats}${_suf}/${dset}"/rttm \
            --wavscp "${data_feats}${_suf}/${dset}"/wav.scp \
            --output_path "${data_feats}${_suf}/${dset}" \
            --sampling_rate "${fs}"
    done
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Remove short data: ${data_feats}/org -> ${data_feats}"

    for dset in "${train_set}" "${valid_set}"; do
    # NOTE: Not applying to test_sets to keep original data

        _spk_list=" "
        _scp_list=" "
        for i in $(seq ${spk_num}); do
            _spk_list+="spk${i} "
            _scp_list+="spk${i}.scp "
        done
        if $use_noise_ref; then
            # references for denoising ("noise1 noise2 ... niose${noise_type_num} ")
            _spk_list+=$(for n in $(seq $noise_type_num); do echo -n "noise$n "; done)
            _scp_list+=$(for n in $(seq $noise_type_num); do echo -n "noise$n.scp "; done)
        fi
        if $use_dereverb_ref; then
            # references for dereverberation
            _spk_list+=$(for n in $(seq $dereverb_ref_num); do echo -n "dereverb$n "; done)
            _scp_list+=$(for n in $(seq $dereverb_ref_num); do echo -n "dereverb$n.scp "; done)
        fi

        # Copy data dir
        utils/copy_data_dir.sh "${data_feats}/org/${dset}" "${data_feats}/${dset}"
        cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"
        for spk in ${_spk_list};do
            cp "${data_feats}/org/${dset}/${spk}.scp" "${data_feats}/${dset}/${spk}.scp"
        done

        _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
        _min_length=$(python3 -c "print(int(${min_wav_duration} * ${_fs}))")

        # utt2num_samples is created by format_wav_scp.sh
        # diarization typically accept long recordings, so does not has
        # max length requirements
        <"${data_feats}/org/${dset}/utt2num_samples" \
            awk -v min_length="${_min_length}" \
                '{ if ($2 > min_length ) print $0; }' \
                >"${data_feats}/${dset}/utt2num_samples"
        for spk in ${_spk_list} "wav"; do
            <"${data_feats}/org/${dset}/${spk}.scp" \
                utils/filter_scp.pl "${data_feats}/${dset}/utt2num_samples"  \
                >"${data_feats}/${dset}/${spk}.scp"
        done

        # fix_data_dir.sh leaves only utts which exist in all files
        #utils/fix_data_dir.sh --utt_extra_files "${_scp_list}" "${data_feats}/${dset}"
        utils/fix_data_dir.sh "${data_feats}/${dset}"
        #sort spk{i}.scp, etc.
        for spk in ${_spk_list} ; do
           sort -t '-' "${data_feats}/${dset}/${spk}.scp" -o "${data_feats}/${dset}/${spk}.scp"
        done            

        # specifics for diarization
        steps/segmentation/convert_utt2spk_and_segments_to_rttm.py \
                "${data_feats}/${dset}"/utt2spk \
                "${data_feats}/${dset}"/segments \
                "${data_feats}/${dset}"/rttm

        # convert standard rttm file into espnet-format rttm (measure with samples)
        pyscripts/utils/convert_rttm.py \
            --rttm "${data_feats}/${dset}"/rttm \
            --wavscp "${data_feats}/${dset}"/wav.scp \
            --output_path "${data_feats}/${dset}" \
            --sampling_rate "${fs}"
    done
fi



# ========================== Data preparation is done here. ==========================