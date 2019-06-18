#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=${1:-0}

PROBLEM=${4:-translate_vndt}
MODEL=${3:-transformer}
HPARAMS=${2:-transformer_base}

PROJECT=$(dirname ${BASH_SOURCE[0]})
T2T_CUSTOM=${PROJECT}/t2t
DATA_DIR=${PROJECT}/t2t_datagen
TMP_DIR=${PROJECT}/input
TRAIN_DIR=${PROJECT}/t2t_train/${PROBLEM}/${MODEL}-${HPARAMS}

# Decode

DECODE_FILE=${TMP_DIR}/test.d

BEAM_SIZE=4
ALPHA=0.6

PREFIX=${PROBLEM}-${MODEL}-${HPARAMS}

t2t-decoder \
  --t2t_usr_dir=${T2T_CUSTOM} \
  --data_dir=${DATA_DIR} \
  --problem=${PROBLEM} \
  --model=${MODEL} \
  --hparams_set=${HPARAMS} \
  --output_dir=${TRAIN_DIR} \
  --decode_hparams="beam_size=${BEAM_SIZE},alpha=${ALPHA}" \
  --decode_from_file=${DECODE_FILE} \
  --decode_to_file=raw-${PREFIX}.pred

paste -d ',' ./input/test.ids raw-${PREFIX}.pred > presub-${PREFIX}.csv
python utils.py post_process --presub=presub-${PREFIX}.csv --sub=sub-${PREFIX}.csv
