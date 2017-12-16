# bleedingedge
Predicting Hematopoietic Cell Lineages from Transcriptomic Data (CS 229 Final Project)

Code by Grace Ng and Anoop Manjunath for CS229 project


MACA_Plate_Notebook.Rmd
In order to grab the remote data and perform the initial pre-processing and normalization, we used existing code provided
by the Chan Zuckerberg Bio-hub project. This code provides the first part of the notebook. Following this, we edited the
remaining sections to perform PCA, and clustering. For these applications we primarily relied on Seurat library (Satija et al., 2015).

neural.py
This file contains the code used to train and run our neural network classifiers, as well as perform k-fold cross validation
for hyperparameter tuning. We relied quite heavily on the scikit-learn database for the creation of this classifier.

softmax_compare_datasets.py
This file contains the code used to compare performance of the softmax regression model on the three data sets (raw, scaled/processed, and
the 34 PCs). Uses the SciKit-Learn Logistic Regression classifier (on the multi-class setting) to perform k-fold
cross validation.

softmax_adjust.py
This file was used to tune hyperparameters for the softmax regression model. Uses the SciKit-Learn Logistic Regression
classifier. Uses the 34 PCs dataset. Randomly sets aside 85% of the data for training/development (using k-fold cross
validation) and the other 15% comprises the test set.

svm_pick_kernel.py
This file was used to select the appropriate kernel for the SVM. Trains 4 classifiers using the SciKit-Learn SVC package.
Tests the performance of the linear, polynomial, radial basis function, and sigmoid kernels on the 34 PCs datset.

svm_linear.py
This file was used to tune hyperparameters on the SVM using a linear kernel. Uses the 34 PCs datset. Even with
hyperparameter tuning, this model performed significantly worse than the SVM with a radial basis kernel. Thus, the
results from this model were not included in the final paper.

svm_rbf.py
This file was used to tune hyperparameters on the SVM using a radial basis kernel. Uses the SciKit-Learn SVC classifier
on the 34 PCs datset. Randomly sets aside 85% of the data for training/development (using k-fold cross validation) and
the other 15% comprises the test set.

