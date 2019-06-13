## Mining Microbial Interactions from Paper Abstracts using Text Classification

**This is the repository for the code for CMPSC 273 Data and Knowledge base (2019 Spring)**

### Repository structure

#### Jupyter notebook for analysis and visualization

`1-data preview.ipynb` contains code for data preview and preprocessing

`2-SVM_res_analysis.ipynb` contains code for BoW+SVM model result analysis

`3-bert_res.ipybn` contains code for BERT model results analysis

#### Bash code for script running

`run_svm.sh` script to run SVM model training

`run_bert.sh` script to train BERT model

`eval_bert.sh` script to only evaluate a trained BERT model

`cross_validation.sh` script to conduct cross validation on BERT model

`cross_validation_continue.sh` script to continue training on cross validation on BERT model. [Still have technique problem and not used in report]

#### Source code to run

`svm_train_adapted.py` python code for BoW+SVM model training

`bert_run_classifier_adapted.py` python code for BERT model training, modified from bert's

