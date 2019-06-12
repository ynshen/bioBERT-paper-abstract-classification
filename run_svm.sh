#!/usr/bin/env bash
export SVM_CLASSIFIER=/home/yuning/Work/cs-273/src/svm_train_adapted.py
export DATA_DIR=/mnt/storage/projects/cs-273/dataset/cross_validation
export OUTPUT=/mnt/storage/projects/cs-273/svm/results_cv_unweighted

python $SVM_CLASSIFIER \
--data_dir=$DATA_DIR \
--output=$OUTPUT \
--C=1 \
--cross_validate=True \
--training=True \
--prediction=True \
--weighted_training=False \
--predict_prob=False
