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

mkdir -p ${TRAIN_DIR}

# Train
# *  If you run out of memory, add --hparams='batch_size=1024' or more.
t2t-trainer \
  --t2t_usr_dir=${T2T_CUSTOM} \
  --data_dir=${DATA_DIR} \
  --problem=${PROBLEM} \
  --model=${MODEL} \
  --hparams_set=${HPARAMS} \
  --output_dir=${TRAIN_DIR}
