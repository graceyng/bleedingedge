import pandas
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import os

#Join the parent directory with the current directory so that we can access files in the parent directory
current_directory = os.path.dirname(__file__)
parent_directory = os.path.split(current_directory)[0]
grandparent_directory = os.path.split(parent_directory)[0]
filenames = ['raw_data', 'raw_data_labels', 'scale_data', 'pcas', 'final_labels']
#filenames = ['raw_data', 'raw_data_labels']
filepaths = {}
for filename in filenames:
    filepaths[filename] = os.path.join(grandparent_directory, filename + '.csv')


csv_files = {}
for filename in filenames:
    if 'labels' in filename:
        csv_files[filename] = pandas.read_csv(filepaths[filename], index_col=0)
    else:
        csv_files[filename] = pandas.read_csv(filepaths[filename], header=0, index_col=0)

print 'finished loading files'

datasets = [{'data': 'raw_data', 'labels': 'raw_data_labels'}, {'data': 'scale_data', 'labels': 'final_labels'},
            {'data': 'pcas', 'labels': 'final_labels'}]



train_accuracies = {'raw_data': [], 'scale_data': [], 'pcas': []}
test_accuracies = {'raw_data': [], 'scale_data': [], 'pcas': []}
avg_train_accuracies = {}
avg_test_accuracies = {}

for dataset in datasets:
    data = dataset['data']
    labels = dataset['labels']
    X = csv_files[data].as_matrix()
    y = np.ravel(csv_files[labels].as_matrix())
    if X.shape[0] != y.shape[0]:
        X = X.T
    M = len(y)

    print labels, y.shape

    """
    #Try 6-fold cross validation instead of 10?
    kf = KFold(n_splits=6, shuffle=True)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        train_data, test_data = X[train_index], X[test_index]
        train_labels, test_labels = y[train_index], y[test_index]
        clf = LogisticRegression(solver='sag', max_iter=100, random_state=42, multi_class='multinomial')
        clf.fit(train_data, train_labels)
        train_predict = clf.predict(train_data)
        train_accuracy = accuracy_score(train_labels, train_predict)
        test_predict = clf.predict(test_data)
        test_accuracy = accuracy_score(test_labels, test_predict)
        print 'train_accuracy for %s: %.4f' % (data, train_accuracy)
        train_accuracies[data].append(train_accuracy)
        print 'test_accuracy for %s: %.4f' % (data, test_accuracy)
        test_accuracies[data].append(test_accuracy)
    avg_train_accuracies = np.mean(train_accuracies[data])
    avg_test_accuracies = np.mean(test_accuracies[data])
    """
