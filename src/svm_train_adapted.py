#!/usr/bin/env python3

__author__ = "Yuning Shen"

"""
Script for SVM classifier, same model from @MInter for new data format and cleaned dataset.
Same flag setup for repeating results	
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
import util
import pdb

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import svm


def boolean_arg(s):
    if s in {'False', 'F', 'false', 'f'}:
        return False
    elif s in {'True', 'T', 'true', 't'}:
        return True
    else:
        raise ValueError('Not a valid boolean string')


class TsvReader(object):

    def __init__(self, text, label):
        self.features = list(text)
        self.labels = label == 1

    @classmethod
    def from_tsv(cls, tsv_dir):
        data = pd.read_csv(tsv_dir, sep='\t')
        return cls(data['text'], data['id'])


class SVMBuilder:

    def __init__(self):
        pass

    @staticmethod
    def get_svm_pipeline(C=1, class_weight=None):
        return Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('svm', svm.SVC(kernel='linear', C=C, class_weight=class_weight, probability=True))
        ])


def main():
    print('main function called')
    print(ARGS.output)
    if not os.path.exists(ARGS.output):
        print(ARGS.output + ' does not exist, create the folder...')
        os.makedirs(ARGS.output)

    if ARGS.weighted_training:
        svm_classifier = SVMBuilder.get_svm_pipeline(C=ARGS.C, class_weight='balanced')
    else:
        svm_classifier = SVMBuilder.get_svm_pipeline(C=ARGS.C)
    print('SVM classifier pipeline created:')
    # print(svm_classifier.get_params())

    if ARGS.cross_validate:
        for cv_set in glob.glob(ARGS.data_dir + '/set*'):
            ix = int(cv_set[-1:])
            print("--------------------------set {}----------------------------".format(ix))
            print("Read data from {}".format(cv_set))
            train = TsvReader.from_tsv(tsv_dir=cv_set + '/train.tsv')
            print('Model training...')
            svm_classifier.fit(train.features, train.labels)
            util.dump_pickle(svm_classifier, ARGS.output + '/cv_model_set_{}'.format(ix),
                             log='SVM model classifier from cross validation set {}'.format(ix),
                             overwrite=True)
            print('\tTraining finished. Model saved to {}'.format(ARGS.output + '/cv_model_set_{}'.format(ix)))
            if ARGS.prediction:
                print('Predict on test set...')
                test = TsvReader.from_tsv(tsv_dir=cv_set + '/dev.tsv')
                if ARGS.predict_prob:
                    prediction = svm_classifier.predict_proba(test.features)
                    # pdb.set_trace()
                    np.savetxt(ARGS.output + '/cv_test_pred_prob_set_{}.txt'.format(ix),
                               prediction,
                               delimiter='\t')
                    print('\tPrediction finished. Result saved to {}'.format(ARGS.output +
                                                                             '/cv_test_pred_prob_set_{}.txt'.format(ix)))
                else:
                    prediction = svm_classifier.predict(test.features)
                    with open(ARGS.output + '/cv_test_pred_set_{}.txt'.format(ix), 'w') as handle:
                        for pred in prediction:
                            handle.write('{}\n'.format(int(pred)))
                    print('\tPrediction finished. Result saved to {}'.format(ARGS.output +
                                                                             '/cv_test_pred_set_{}.txt'.format(ix)))
    else:
        print("Import training data...")
        if ARGS.training_data is not None:
            train = TsvReader.from_tsv(tsv_dir=ARGS.training_data)
        else:
            train = TsvReader.from_tsv(tsv_dir=ARGS.data_dir + '/train.tsv')
        # pdb.set_trace()
        print('Model training...')
        svm_classifier.fit(train.features, train.labels)
        util.dump_pickle(svm_classifier, ARGS.output + '/model_trained.pkl',
                         log='SVM model classifier from training set.',
                         overwrite=True)
        print('\tTraining finished. Model saved to {}'.format(ARGS.output + '/model_trained.pkl'))
        if ARGS.prediction:
            print('Predict on test set...')
            test = TsvReader.from_tsv(tsv_dir=ARGS.data_dir + '/test.tsv')
            # pdb.set_trace()
            if ARGS.predict_prob:
                prediction = svm_classifier.predict_proba(test.features)
                np.savetxt(ARGS.output + '/test_pred_prob.txt',
                           prediction,
                           delimiter='\t')
                print('\tPrediction finished. Result saved to {}'.format(ARGS.output + '/test_pred_prob.txt'))
        else:
            prediction = svm_classifier.predict(test.features)
            with open(ARGS.output + '/test_pred.txt', 'w') as handle:
                for pred in prediction:
                    handle.write('{}\n'.format(int(pred)))
            print('\tPrediction finished. Result saved to {}'.format(ARGS.output + '/test_pred.txt'))

if __name__ == "__main__":
    print('Running')
    parser = argparse.ArgumentParser(description="Support Vector Machine classifier adapted from @MInter")
    parser.add_argument("-d", "--data_dir", help ="Directory to data .tsv file for training/validation")
    parser.add_argument("-o", "--output", help ="Output path for pickled SVM or training/prediction results")
    parser.add_argument("-C", "--C", help = "C value of SVM", type = int, default = 1)
    parser.add_argument("-c", '--cross_validate', type = boolean_arg, default = False)
    parser.add_argument("-t", '--training', type = boolean_arg, default = True)
    parser.add_argument("-T", '--training_data', type = str, help="Set a .tsv file for training", default = None)
    parser.add_argument("-p", '--prediction', type = boolean_arg, default = True)
    parser.add_argument("-w", '--weighted_training', type = boolean_arg, default = True)
    parser.add_argument("-P", "--predict_prob", type = boolean_arg, default=False)
    ARGS = parser.parse_args()
    print("Arguements:\n{}".format(ARGS))
    main()