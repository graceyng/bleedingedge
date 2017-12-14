# bleedingedge
Predicting Hematopoietic Cell Lineages from Transcriptomic Data (CS 229 Final Project)

Code by Grace Ng and Anoop Manjunath for CS229 project


MACA_Plate_Notebook.Rmd
In order to grab the remote data and perform the initial pre-processing and normalization, we used existing code provided by the Chan Zuckerberg Bio-hub project. This code provides the first part of the notebook. Following this, we edited the remaining sections to perform PCA, and clustering. For these applications we primarily relied on Seurat library (Satija et al., 2015).

neural.py
This file contains the code used to train and run our neural network classifiers, as well as perform k-fold cross validation for hyperparameter tuning. We relied quite heavily on the scikit-learn database for the creation of this classifier

log_regression.py

