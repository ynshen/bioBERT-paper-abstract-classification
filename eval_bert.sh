export BERT_SCRIPT=/home/yuning/Work/cs-273/src/ref/bert
export BERT_BASE_DIR=/mnt/storage/projects/cs-273/bert_pretrained/uncased_L-12_H-768_A-12
export TRAINED_MODEL=/mnt/storage/projects/cs-273/bert_test/tmp/model.ckpt-829

python $BERT_SCRIPT/run_classifier.py \
--task_name=cola \
--do_train=false \
--do_eval=false \
--do_predict=true \
--data_dir=/mnt/storage/projects/cs-273/bert_test \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$TRAINED_MODEL \
--max_seq_length=512 \
--train_batch_size=4 \
--learning_rat=2e-5 \
--num_train_epochs=5.0 \
--output_dir=/mnt/storage/projects/cs-273/bert_test/tmp/predict

