#==============================================================================
# ###########################################################
#                                                  
#  The following file is used to initialise pre-CNN models 
# 
#  The user has nine models to choose from:
#  - SGD Classifier (Linear models with Stochastic Gradient Descent)
#  - SVM (Linear Kernel)
#  - Logistic Regression
#  - K Nearest Neighbours
#  - Random Forest
#  - Gradient Boosting
#  - AdaBoost
#  - Ridge Classifier
#  - Extra Trees
#
#  The user then has three methods to choose from:
#   - tuned model using 100% training data
#   - tuned model using 10-fold cross validation on training data
#   - grid search using 10-fold cross validation on training data
#
# ###########################################################
#==============================================================================

import sys, os
from time import time
from evaluation import output
from classifiers import sklearn_classifiers as skc

startup_text = ('\nWelcome to Group 34\'s code \n'
+'\n We have a range of models to choose from:'
+'\n------------------------------------------------------------------------------------'
+'\n|Option|  Report |     Model    |      Classifier      | ~Accuracy  | Timing 8GB   |'
+'\n| (1)  |   Yes   |    Linear    | SGD Classifier       |   0.3721   | ~0.5 minutes |'  
+'\n| (2)  |   Yes   |    Linear    | SVM (Linear Kernel)  |   0.4034   | ~3.5 minutes |'  
+'\n| (3)  |   Yes   |    Linear    | Logistic Regression  |   0.4174   | ~2.5 minutes |'        
+'\n| (4)  |   Yes   |  Non-linear  | Nearest Neighbours   |   0.4195   | ~4.0 minutes |'
+'\n| (5)  |   Yes   |  Non-linear  | Random Forest        |   0.4314   | ~2.5 minutes |'
+'\n| (6)  |   Yes   |  Non-linear  | Gradient Boosting    |   0.5132   | ~50 minutes  |'
+'\n| (7)  |   No    |  Non-linear  | AdaBoost             |   0.3701   | ~5.0 minutes |'
+'\n| (8)  |   No    |    Linear    | Ridge Classifier     |   0.3964   | ~2.5 minutes |'
+'\n| (9)  |   No    |  Non-linear  | Extra Trees          |   0.4343   | ~2.5 minutes |'
+'\n------------------------------------------------------------------------------------'
+'\n\nInput your selection as a number: ')

method_text = ('\n We have a range of methods to choose from:'
+'\n---------------------------------------------------------------------------'
+'\n|Option|    Method    |         Data              |        Timing 8GB     |'
+'\n| (1)  | Final tuned  | Using 100% train data     |    As above           |'  
+'\n| (2)  | Final tuned  | Using 10-fold CV on train |    As above per fold  |'  
+'\n| (3)  | Grid Search  | Using 10-fold CV on train |    ~ 1 - 5 days       |'
+'\n---------------------------------------------------------------------------'
+'\n\nInput your selection as a number: ')

file_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(file_directory)
input_directory = os.path.join(parent_directory, 'input/')
output_directory = os.path.join(parent_directory, 'output/')

sys.path.append(file_directory)
ofile, wfile = output.prepare_file(output_directory, 'sklearn_results.csv')

paths = {}

paths['confusion_matrix'] = os.path.join(output_directory, 'confusion_matrix.xlsx').replace('\\', '/')

start = time()
model = int(input(startup_text))
method = int(input(method_text))

if model > 9 or method > 3:
    print("\nInvalid selection\n--Exiting Program---")

elif model == 1:
    skc.sgd_classifier(wfile, paths, method).train()

elif model == 2:
    skc.linear_svc(wfile, paths, method).train()

elif model == 3:
    skc.logistic_regression(wfile, paths, method).train()

elif model == 4:
    skc.nearest_neighbours(wfile, paths, method).train()
    
elif model == 5:
    skc.random_forest(wfile, paths, method).train()
    
elif model == 6:
    skc.gradient_boost(wfile, paths, method).train()
    
elif model == 7:
    skc.ada_boost(wfile, paths, method).train()
    
elif model == 8:
    skc.ridge_classifier(wfile, paths, method).train()
    
elif model == 9:
    skc.extra_trees(wfile, paths, method).train()
    
end = time() - start
output.close_file(ofile)
print('Total time to run %0.3fs' %end)
print('\n---DONE---')

