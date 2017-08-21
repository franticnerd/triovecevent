import operator
import sys

import numpy as np
import pandas as pd
from numpy.random import permutation
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from zutils.config.param_handler import yaml_loader


def load_train_test(feature_file, label_file):
    # 1. first load all the features and labels from the files
    features = pd.read_csv(feature_file, sep='\t')
    candidate_id_labels = []
    with open(label_file, 'r') as fin:
        for line in fin:
            items = line.strip().split('\t')
            candidate_id, label = int(items[0]), int(items[1])
            candidate_id_labels.append((candidate_id, label))
    candidate_id_labels.sort(key=operator.itemgetter(0), reverse=False)
    labels = [e[1] for e in candidate_id_labels]
    # features, labels = shuffle(features, labels)

    # 2. Use the first 50% as the training data, and the rest 50% as test data
    n_train = int(len(labels) * 0.5)
    features_train = features.head(n_train)
    labels_train = labels[:n_train]
    features_test = features.tail(len(labels) - n_train)
    labels_test = labels[n_train:]

    # 3. feature normalization
    standard_scaler = StandardScaler()
    features_train = standard_scaler.fit_transform(features_train)
    features_test = standard_scaler.transform(features_test)

    # 4. downsampling
    # features_train, labels_train = subsample(features_train, labels_train)
    # features_test, labels_test = subsample(features_test, labels_test)

    return features_train, labels_train, features_test, labels_test


def shuffle(features, labels):
    rand_index = permutation(len(labels))
    shuffled_features, shuffled_labels = [], []
    for i in rand_index:
        shuffled_labels.append(labels[i])
    shuffled_features = features.iloc[rand_index]
    return shuffled_features, shuffled_labels


def subsample(f_train, y_train):
    # np.random.seed(100)
    major_index, minor_index = [], []
    for i, y in enumerate(y_train):
        if y == 0:
            major_index.append(i)
        else:
            minor_index.append(i)
    rand_index = permutation(len(major_index))
    result_features = []
    result_labels = []
    # put in minor examples
    for i in minor_index:
        result_features.append(f_train[i])
        result_labels.append(y_train[i])
    # put in major examples
    for i in rand_index[:len(minor_index)]:
        index = major_index[i]
        result_features.append(f_train[index])
        result_labels.append(y_train[index])
    return result_features, result_labels


def train(features, labels):
    # Random Forest Classifier
    model = RandomForestClassifier()
    model.fit(features, labels)
    print model.feature_importances_

    model = LogisticRegression()
    # model.fit(features, labels)
    # print feature importance
    rfe = RFE(model, 1)
    fit = rfe.fit(features, labels)
    # print("Num Features: %d") % fit.n_features_
    # print("Selected Features: %s") % fit.support_
    print("Feature Ranking: %s") % fit.ranking_
    
    model.fit(features, labels)
    return model


def eval(model, features, labels):
    expected = labels
    predicted = model.predict(features)
    prfs = precision_recall_fscore_support(expected, predicted)
    precision, recall, f1 = prfs[0][1], prfs[1][1], prfs[2][1]
    return precision, recall, f1


def run(feature_file, label_file):
    features_train, labels_train, features_test, labels_test = load_train_test(feature_file, label_file)
    model = train(features_train, labels_train)
    return eval(model, features_test, labels_test)
    # print accuracy, micro_f1, macro_f1, weighted_f1


if __name__ == '__main__':
    # random.seed(100)
    data_dir = '/Users/chao/Dropbox/data/class_event/ny/'
    if len(sys.argv) > 1:
        para_file = sys.argv[1]
        para = yaml_loader().load(para_file)
        data_dir = para['data_dir']
    feature_file = data_dir + 'classify_candidate_features.txt'
    label_file = data_dir + 'classify_candidate_labels.txt'
    performance_file = data_dir + 'classify_candidate_performance.txt'
    precisions, recalls, fscores = [], [], []
    for i in xrange(100):
        p, r, f = run(feature_file, label_file)
        precisions.append(p)
        recalls.append(r)
        fscores.append(f)
    precision, recall, fscore = np.mean(precisions), np.mean(recalls), np.mean(fscores)
    print precision, recall, fscore
    with open(performance_file, 'w') as fout:
        fout.write('\t'.join(['Method', 'Precision', 'Recall', 'F1']) + '\n')
        fout.write('\t'.join(['ClassEvent', str(precision), str(recall), str(fscore)]) + '\n')
