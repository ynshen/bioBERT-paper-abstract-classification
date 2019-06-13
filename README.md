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

`src/svm_train_adapted.py` python code for BoW+SVM model training

`src/bert_run_classifier_adapted.py` python code for BERT model training, modified from BERT's `run_classifier.py` add function of 
 - Tracking training process
 - Option to conduct cross validation
 - Option to use weighted loss function

`src/preprocesssing.py`, `src/util.py`, `src/test.py`: some other utility functions for the project

#### Training experiment log
- Saved under `exp_logs`

#### Figures
- Saved under `figs`

#### Dependent packages
Under `src/ref`, for unknown reason, these folders are not be able to upload, here are links to the original repo:
- [atmister](https://github.com/CSB5/atminter)
- [bert](https://github.com/google-research/bert)

#### Trained models and data
Due to storage constrains, trained models and data are still stored off-line on local machine. They will be provided per request to yuningshen@ucsb.edu

