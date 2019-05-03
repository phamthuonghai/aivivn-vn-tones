#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3

PROBLEM=translate_vndt
MODEL=transformer
HPARAMS=transformer_base

PROJECT=$(dirname ${BASH_SOURCE[0]})
T2T_CUSTOM=${PROJECT}/t2t
DATA_DIR=${PROJECT}/t2t_datagen
TMP_DIR=${PROJECT}/input
TRAIN_DIR=${PROJECT}/t2t_train/${PROBLEM}/${MODEL}-${HPARAMS}

# Decode

DECODE_FILE=${TMP_DIR}/test.d

BEAM_SIZE=4
ALPHA=0.6

t2t-decoder \
  --t2t_usr_dir=${T2T_CUSTOM} \
  --data_dir=${DATA_DIR} \
  --problem=${PROBLEM} \
  --model=${MODEL} \
  --hparams_set=${HPARAMS} \
  --output_dir=${TRAIN_DIR} \
  --decode_hparams="beam_size=${BEAM_SIZE},alpha=${ALPHA}" \
  --decode_from_file=${DECODE_FILE} \
  --decode_to_file=test_toned.pred

paste -d ',' ./input/test.ids test_toned.pred > presub.txt
python utils.py post_process
