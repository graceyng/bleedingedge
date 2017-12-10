import pandas
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import os

#Join the parent directory with the current directory so that we can access files in the parent directory
current_directory = os.path.dirname(__file__)
parent_directory = os.path.split(current_directory)[0]
grandparent_directory = os.path.split(parent_directory)[0]
filenames = ['34pcas', 'final_labels']
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

dataset = {'data': '34pcas', 'labels': 'final_labels'}

kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']
train_accuracies = {kernel: [] for kernel in kernel_list}
test_accuracies = {kernel: [] for kernel in kernel_list}
"""
validate_accuracies = {kernel: [] for kernel in kernel_list}
"""

data = dataset['data']
labels = dataset['labels']
X = csv_files[data].as_matrix()
y = np.ravel(csv_files[labels].as_matrix())
if X.shape[0] != y.shape[0]:
    X = X.T
M = len(y)
print X.shape
print y.shape
"""
p = np.random.permutation(M)
perm_X = X[p]
perm_y = y[p]
num_test_ex = int(np.floor(0.15 * M))
test_data = X[0:num_test_ex]
test_labels = y[0:num_test_ex]
train_validate_data = X[num_test_ex:]
train_validate_labels = y[num_test_ex:]
"""

"""
#Try 6-fold cross validation instead?
kf = KFold(n_splits=10, shuffle=True)
kf.get_n_splits(X)
for train_index, test_index in kf.split(X):
    for kernel in kernel_list:
        train_data, test_data = X[train_index], X[test_index]
        train_labels, test_labels = y[train_index], y[test_index]
        clf = svm.SVC(kernel=kernel, max_iter=200)
        clf.fit(train_data, train_labels)
        train_predict = clf.predict(train_data)
        train_accuracy = accuracy_score(train_labels, train_predict)
        test_predict = clf.predict(test_data)
        test_accuracy = accuracy_score(test_labels, test_predict)
        print 'train_accuracy for %s: %.4f' % (kernel, train_accuracy)
        train_accuracies[kernel].append(train_accuracy)
        print 'test_accuracy for %s: %.4f' % (kernel, test_accuracy)
        test_accuracies[kernel] = test_accuracy

avg_train_accuracies = {kernel: np.mean(train_accuracies[kernel]) for kernel in kernel_list}
avg_test_accuracies = {kernel: np.mean(test_accuracies[kernel]) for kernel in kernel_list}
"""

"""
#Try 6-fold cross validation instead?
kf = KFold(n_splits=6, shuffle=True)
kf.get_n_splits(train_validate_data)
for train_index, validate_index in kf.split(train_validate_data):
    for kernel in kernel_list:
        train_data, validate_data = train_validate_data[train_index], train_validate_data[validate_index]
        train_labels, validate_labels = train_validate_labels[train_index], train_validate_labels[validate_index]
        clf = svm.SVC(kernel=kernel, max_iter=200)
        clf.fit(train_data, train_labels)
        train_predict = clf.predict(train_data)
        train_accuracy = accuracy_score(train_labels, train_predict)
        validate_predict = clf.predict(validate_data)
        validate_accuracy = accuracy_score(validate_labels, validate_predict)
        print 'train_accuracy for %s: %.4f' % (kernel, train_accuracy)
        train_accuracies[kernel].append(train_accuracy)
        print 'validate_accuracy for %s: %.4f' % (kernel, validate_accuracy)
        validate_accuracies[kernel] = validate_accuracy

mean_train_accuracies = [np.mean(np.array(train_accuracies[kernel])) for kernel in kernel_list]
mean_validate_accuracies = [np.mean(np.array(validate_accuracies[kernel])) for kernel in kernel_list]
best_kernel_ind = np.argmax(mean_validate_accuracies)
print 'Best validation performance: kernel %s' %kernel_list[best_kernel_ind]
clf = svm.SVC(kernel=kernel_list[best_kernel_ind], max_iter=200)
clf.fit(train_validate_data, train_validate_labels)
train_validate_predict = clf.predict(train_validate_data)
train_validate_accuracy = accuracy_score(train_validate_labels, train_validate_predict)
print 'Final training accuracy: %.4f' %train_validate_accuracy
test_predict = clf.predict(test_data)
test_accuracy = accuracy_score(test_labels, test_predict)
print 'Test accuracy: %.4f' %test_accuracy"""