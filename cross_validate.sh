#!/usr/bin/env bash
export BERT_BASE_DIR=/mnt/storage/projects/cs-273/bert_pretrained/biobert_v1.1_pubmed
export CHECKPOINT_NAME=model.ckpt-1000000
export BERT_SCRIPT=/home/yuning/Work/cs-273/src/bert_run_classifier_adapted.py
for SET in 1 2 3 4 5
do
    echo "------------------------------Processing Set_$SET-----------------------------------"
    python $BERT_SCRIPT \
    --task_name=stc \
    --do_train=true \
    --do_eval=true \
    --do_predict=false \
    --data_dir=/mnt/storage/projects/cs-273/dataset/cross_validation/set_$SET \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/$CHECKPOINT_NAME \
    --max_seq_length=512 \
    --train_batch_size=4 \
    --log_step_interval=50 \
    --save_checkpoints_steps=100 \
    --learning_rate=2e-5 \
    --num_train_epochs=8.0 \
    --output_dir=/mnt/storage/projects/cs-273/bert_cv/BioBert_8_epoch_weighted/set_$SET \
    --silent_example=true \
    --positive_percent=.07459 \
    --cross_validate=true
done