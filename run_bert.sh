export BERT_BASE_DIR=/mnt/storage/projects/cs-273/bert_pretrained/uncased_L-12_H-768_A-12
export CHECKPOINT_NAME=bert_model.ckpt
export BERT_SCRIPT=/home/yuning/Work/cs-273/src/bert_run_classifier_adapted.py

python $BERT_SCRIPT \
--task_name=stc \
--do_train=true \
--do_eval=true \
--do_predict=true \
--data_dir=/mnt/storage/projects/cs-273/bert_test \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/$CHECKPOINT_NAME \
--max_seq_length=512 \
--train_batch_size=4 \
--log_step_interval=25 \
--save_checkpoint_steps=200 \
--learning_rat=2e-5 \
--num_train_epochs=1.0 \
--output_dir=/mnt/storage/projects/cs-273/bert_res/Bert_1_epochs_test \
--silent_example=true
