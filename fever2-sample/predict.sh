#!/usr/bin/env bash

CUDA_DEVICE=0
default_cuda_device=0
root_dir=/users/k21190024/study/fact-checking-repos/fever/sheffieldnlp/fever2-sample
staging=$root_dir/data/predstage

# ln -s $root_dir/data data

echo "start evidence retrieval"

python -m fever.evidence.retrieve \
    --index $root_dir/data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
    --database $root_dir/data/fever/fever.db \
    --in-file $1 \
    --out-file $staging/ir.$(basename $1) \
    --max-page 5 \
    --max-sent 5 \
    --parallel True \
    --threads 25

echo "start prediction"
python -m allennlp.run predict \
    $root_dir/data/models/decomposable_attention.tar.gz \
    $staging/ir.$(basename $1) \
    --output-file $staging/labels.$(basename $1) \
    --predictor fever \
    --include-package fever.reader \
    --cuda-device ${CUDA_DEVICE:-$default_cuda_device} \
    --silent

echo "prepare submission"
python -m fever.submission.prepare \
    --predicted_labels $staging/labels.$(basename $1) \
    --predicted_evidence $staging/ir.$(basename $1) \
    --out_file $2
