#!/usr/bin/env bash

CUDA_DEVICE=-1
default_cuda_device=0

model_path=$2

root_dir=$1
staging=$root_dir/predstage

mkdir -p $staging

echo "start prediction"
python -m allennlp.run predict \
    $model_path \
    $3 \
    --output-file $staging/labels.$(basename $3) \
    --predictor fever \
    --include-package fever.reader \
    --cuda-device ${CUDA_DEVICE:-$default_cuda_device} \
    --overrides '{"dataset_reader": {"database": "/users/k21190024/study/fact-checking-repos/fever/baseline/dumps/feverised-scifact/feverised-scifact-titleid.db"}}' \
    --silent

echo "prepare submission"
python -m fever.submission.prepare \
    --predicted_labels $staging/labels.$(basename $3) \
    --predicted_evidence $3 \
    --out_file $root_dir/pred.$(basename $3)
