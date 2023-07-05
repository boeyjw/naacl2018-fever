#!/usr/bin/env bash

CUDA_DEVICE=-1
default_cuda_device=0
model_path=$5

root_dir=$1
staging=$root_dir/predstage

index_path=$2
database_path=$3

mkdir -p $staging

# ln -s $root_dir/data data

echo "start evidence retrieval"

python -m fever.evidence.retrieve \
    --index $index_path \
    --database  $database_path\
    --in-file $4 \
    --out-file $staging/ir.$(basename $4) \
    --max-page 5 \
    --max-sent 5 \
    --parallel True \
    --threads 25

echo "start prediction"
# May have to change database path in library file /scratch/users/k21190024/envs/conda/fever-baseline/lib/python3.6/site-packages/fever/reader/fever_reader.py to db associated with the data

python -m allennlp.run predict \
    $model_path \
    $staging/ir.$(basename $4) \
    --output-file $staging/labels.$(basename $4) \
    --predictor fever \
    --include-package fever.reader \
    --cuda-device ${CUDA_DEVICE:-$default_cuda_device} \
    --overrides '{"dataset_reader": {"database": "/users/k21190024/study/fact-checking-repos/fever/baseline/dumps/feverised-climatefever/feverised-climatefever-titleid.db"}}' \
    --silent

echo "prepare submission"
python -m fever.submission.prepare \
    --predicted_labels $staging/labels.$(basename $4) \
    --predicted_evidence $staging/ir.$(basename $4) \
    --out_file $root_dir/pred.$(basename $4)
