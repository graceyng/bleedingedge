import pandas
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
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

penalty_types = [['l2', 'sag'], ['l1', 'saga']]
C = [0.001, 0.005, 0.008, 0.01, 0.02, 0.05, 0.1, 0.5, 1., 1.5, 2.] #smaller values of C corespond to more weight on the regularization term

params_to_try = []
for penalty_type in penalty_types:
        for c in C:
            params_to_try.append({'penalty_type': penalty_type[0], 'solver': penalty_type[1], 'C': c})

train_accuracies = [[] for i in range(len(params_to_try))]
train_auroc = [[] for i in range(len(params_to_try))]
validate_accuracies = [[] for i in range(len(params_to_try))]
validate_auroc = [[] for i in range(len(params_to_try))]

data = dataset['data']
labels = dataset['labels']
X = csv_files[data].as_matrix()
y = np.ravel(csv_files[labels].as_matrix())
if X.shape[0] != y.shape[0]:
    X = X.T
M = len(y)

np.random.seed(42)
p = np.random.permutation(M)
perm_X = X[p]
perm_y = y[p]
num_test_ex = int(np.floor(0.15 * M))
test_data = X[0:num_test_ex]
test_labels = y[0:num_test_ex]
train_validate_data = X[num_test_ex:]
train_validate_labels = y[num_test_ex:]

#Try 6-fold cross validation instead?
kf = KFold(n_splits=6, shuffle=True)
kf.get_n_splits(train_validate_data)
for train_index, validate_index in kf.split(train_validate_data):
    for i, params in enumerate(params_to_try):
        train_data, validate_data = train_validate_data[train_index], train_validate_data[validate_index]
        train_labels, validate_labels = train_validate_labels[train_index], train_validate_labels[validate_index]
        clf = LogisticRegression(penalty=params['penalty_type'], C=params['C'], solver=params['solver'], max_iter=100,
                                 random_state=42, multi_class='multinomial')
        clf.fit(train_data, train_labels)
        train_predict = clf.predict(train_data)
        train_accuracy = accuracy_score(train_labels, train_predict)
        validate_predict = clf.predict(validate_data)
        validate_accuracy = accuracy_score(validate_labels, validate_predict)
        #print 'train_accuracy for %s: %.4f' % (kernel, train_accuracy)
        train_accuracies[i].append(train_accuracy)
        train_auroc[i].append(roc_auc_score(train_labels, train_predict))
        #print 'validate_accuracy for %s: %.4f' % (kernel, validate_accuracy)
        validate_accuracies[i].append(validate_accuracy)
        validate_auroc[i].append(roc_auc_score(validate_labels, validate_predict))

mean_train_accuracies = [np.mean(np.array(train_accuracies[i])) for i in range(len(params_to_try))]
mean_validate_accuracies = [np.mean(np.array(validate_accuracies[i])) for i in range(len(params_to_try))]
mean_train_auroc = [np.mean(np.array(train_auroc[i])) for i in range(len(params_to_try))]
mean_validate_auroc = [np.mean(np.array(validate_auroc[i])) for i in range(len(params_to_try))]

best_accuracy_ind = np.argmax(mean_validate_accuracies)
best_auroc_ind = np.argmax(mean_validate_auroc)
train_combined_metric = 0.6 * np.array(mean_validate_accuracies) + 0.4*np.array(mean_validate_auroc)
best_combined_ind = np.argmax(train_combined_metric)

best_indexes = {'accuracy': best_accuracy_ind, 'AUROC': best_auroc_ind, 'combined': best_combined_ind}
for index_type, best_index in best_indexes.iteritems():
    params = params_to_try[best_index]
    print 'Best validation %s: penalty type %s, C-value %.2f' %(index_type, params['penalty_type'], params['C'])
    clf = LogisticRegression(penalty=params['penalty_type'], C=params['C'], solver=params['solver'], max_iter=100,
                             random_state=42, multi_class='multinomial')
    clf.fit(train_validate_data, train_validate_labels)
    train_validate_predict = clf.predict(train_validate_data)
    train_validate_accuracy = accuracy_score(train_validate_labels, train_validate_predict)
    print 'Final training accuracy for best validation accuracy: %.4f' %train_validate_accuracy
    test_predict = clf.predict(test_data)
    test_accuracy = accuracy_score(test_labels, test_predict)
    print 'Test accuracy: %.4f' %test_accuracy
    print 'Test AUROC: %.4f' %roc_auc_score(test_labels, test_predict)
