# sklearn_cifar10

This repo uses sklearn and gridsearch to discover the best performing pre-CNN algorithms on the CIFAR10 dataset.

The following algorithms have been used:
 - Logistic Regression
 - Ridge Regression
 - SGD Classifier
 - Linear SVC (SVM with Linear Kernel)
 - K Nearest Neighbours
 - Random Forest
 - Extremely Randomized Trees (Extra Trees)
 - Gradient Boosting
 - Adaptive Boosting (AdaBoost)

Each classifer can be trained with a number of options:
 - PCA / SVD dimensionality reduction
 - 10-fold cross validation on training data
 - 100% of training data used and evaluated on test data
 - grid search
 - write evaluation metrics to a csv
 - write confusion matrix to a xlsx
