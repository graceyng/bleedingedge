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

data = dataset['data']
labels = dataset['labels']
X = csv_files[data].as_matrix()
y = np.ravel(csv_files[labels].as_matrix())
if X.shape[0] != y.shape[0]:
    X = X.T
M = len(y)

gammas = np.array([.5, .8, 1., 1.1, 1.5, 2., 5., 10.]) / float(M)
C = [1, 2., 2.5, 3., 3.5, 4., 5., 10., 30.] #smaller values of C corespond to more weight on the regularization term

params_to_try = []
for gamma in gammas:
        for c in C:
            params_to_try.append({'gamma': gamma, 'C': c})

train_accuracies = [[] for i in range(len(params_to_try))]
validate_accuracies = [[] for i in range(len(params_to_try))]

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
        clf = svm.SVC(C=params['C'], gamma=params['gamma'], kernel='rbf', max_iter=200)
        clf.fit(train_data, train_labels)
        train_predict = clf.predict(train_data)
        train_accuracy = accuracy_score(train_labels, train_predict)
        validate_predict = clf.predict(validate_data)
        validate_accuracy = accuracy_score(validate_labels, validate_predict)
        #print 'train_accuracy for %s: %.4f' % (kernel, train_accuracy)
        train_accuracies[i].append(train_accuracy)
        #print 'validate_accuracy for %s: %.4f' % (kernel, validate_accuracy)
        validate_accuracies[i].append(validate_accuracy)

mean_train_accuracies = [np.mean(np.array(train_accuracies[i])) for i in range(len(params_to_try))]
mean_validate_accuracies = [np.mean(np.array(validate_accuracies[i])) for i in range(len(params_to_try))]
best_ind = np.argmax(mean_validate_accuracies)
print 'Best validation performance: gamma %.2f, C-value %.2f' %(params_to_try[best_ind]['gamma'], params_to_try[best_ind]['C'])
clf = svm.SVC(C=params['C'], gamma=params['gamma'], kernel='rbf', max_iter=200)
clf.fit(train_validate_data, train_validate_labels)
train_validate_predict = clf.predict(train_validate_data)
train_validate_accuracy = accuracy_score(train_validate_labels, train_validate_predict)
print 'Final training accuracy: %.4f' %train_validate_accuracy
test_predict = clf.predict(test_data)
test_accuracy = accuracy_score(test_labels, test_predict)
print 'Test accuracy: %.4f' %test_accuracy