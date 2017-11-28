#==============================================================================
# ###########################################################
#                                                  
#  The following file is used to train pre-CNN models used during exploration and grid search
# 
#  The below classifiers have been chosen:
#  - Logistic Regression
#  - Ridge Regression
#  - SGD Classifier
#  - Linear SVC (SVM with Linear Kernel)
#  - K Nearest Neighbours
#  - Random Forest
#  - Extremely Randomized Trees (Extra Trees)
#  - Gradient Boosting
#  - Adaptive Boosting (AdaBoost)
#
#  Each class enables users to:
#  - 10-fold cross validation on training data
#  - 100% of training data used and evaluated on test data
#  - grid search
#  - write evaluation metrics to a csv
#  - write confusion matrix to a xlsx
#
# ###########################################################
#==============================================================================

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from time import time
from evaluation import output
from evaluation import metrics
from preprocessing import data

class logistic_regression:
    
    def __init__(self, file, paths, method):
        
        self.file = file
        self.paths = paths
        self.output_dict = {}
        self.method = method
        
    def create_dict(self):
        
        self.output_dict = output.init_dict()
        self.output_dict['model'] = 'Logistic Regression'
        
    def write(self):
        
        output.write_row(self.file, self.output_dict)
        
    def train(self):
        print('\n--- Running Logistic Regression ---')
        if self.method == 1:
            print('-- Tuned model on all 50000 samples --')
        if self.method == 2:
            print('-- Tuned model using 10-fold cross validation --')
        if self.method == 3:
            print('-- Grid search using 10-fold cross validation --')
            
        X_train, X_test, y_train, y_test = data.get_data()
        self.create_dict()
        
        if self.method == 1:
            
            self.optimal(X_train, X_test, y_train, y_test)
        
        elif self.method == 2 or self.method == 3:
                
            kth_fold = 1
        
            for train_index, test_index in KFold(n_splits = 10).split(X_train):
                    
                X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
                y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
                
                print('\n-- Cross Validation Fold: %i --' %kth_fold)
                          
                if self.method == 2:
                    
                    self.output_dict['validation'] = kth_fold
                    self.optimal(X_train_cv, X_test_cv, y_train_cv, y_test_cv)
                    
                if self.method == 3:
                    
                    self.gridsearch(X_train_cv, X_test_cv, y_train_cv, y_test_cv)
                             
                kth_fold += 1
                         
    def optimal(self, X_train, X_test, y_train, y_test):
        
        self.output_dict['hype_penalty'] = 'l2'
        self.output_dict['hype_c'] = 0.01

        pc = 1000
 
        if pc != X_train.shape[1]:
            start = time()
            svd = PCA(n_components=pc, svd_solver='full')
            X_train = svd.fit_transform(X_train)
            X_test = svd.transform(X_test)
            process_time = time() - start
            
            self.output_dict['preprocess_time'] = process_time
            self.output_dict['retained_variance'] = np.sum(svd.explained_variance_ratio_)
       
        self.predict(X_train, X_test, y_train, y_test)
        
    def gridsearch(self, X_train, X_test, y_train, y_test):
        
        hype_penalty = ['l2']
        hype_c = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 2, 3, 5]
        components = [10, 20, 30, 50, 100, 200, 500, 1000, 2000, X_train.shape[1]]
        
        for pc in components:
            
            if pc != X_train.shape[1]:
                start = time()
                svd = PCA(n_components=pc, svd_solver='full')
                X_train_r = svd.fit_transform(X_train)
                X_test_r = svd.transform(X_test)                
                self.output_dict['preprocess_time'] = time() - start
                self.output_dict['retained_variance'] = np.sum(svd.explained_variance_ratio_)
            else:
                X_train_r = X_train
                X_test_r = X_test
                self.output_dict['preprocess_time'] = 0
                self.output_dict['retained_variance'] = 1
            
            for c in hype_c:
                
                self.output_dict['hype_c'] = c
                                
                for p in hype_penalty:
                    
                    self.output_dict['hype_penalty'] = p
                    self.predict(X_train_r, X_test_r, y_train, y_test)
        
    def predict(self, X_train, X_test, y_train, y_test):
        
        self.output_dict['dimensions'] = X_train.shape[1]
        self.output_dict['observations'] = X_train.shape[0]
                
        start = time()
        clf = LogisticRegression(
                penalty = self.output_dict['hype_penalty'], 
                C = self.output_dict['hype_c'], 
                multi_class='multinomial', 
                solver='lbfgs')
        mdl = clf.fit(X_train, y_train)
        y_hat = mdl.predict(X_train)
        
        self.output_dict['train_time'] = time() - start
        self.output_dict['train_accuracy'] = accuracy_score(y_hat, y_train)
        self.output_dict['train_precision'] = precision_score(y_hat, y_train, average='weighted')
        self.output_dict['train_recall'] = recall_score(y_hat, y_train, average='weighted')
        self.output_dict['train_fscore'] = f1_score(y_hat, y_train, average='weighted')
    
        start = time()
        y_hat = mdl.predict(X_test)

        self.output_dict['test_time'] = time() - start
        self.output_dict['test_accuracy'] = accuracy_score(y_hat, y_test)
        self.output_dict['test_precision'] = precision_score(y_hat, y_test, average='weighted')
        self.output_dict['test_recall'] = recall_score(y_hat, y_test, average='weighted')
        self.output_dict['test_fscore'] = f1_score(y_hat, y_test, average='weighted')
        self.output_dict['total_time'] = self.output_dict['train_time'] + self.output_dict['test_time'] + self.output_dict['preprocess_time']
        
        self.write()
        
        if self.method == 1 or self.method == 2:
            
            print('Time to process: %0.6f' %self.output_dict['preprocess_time'])
            print('Time to train: %0.6f' %self.output_dict['train_time'])
            print('Time to test: %0.6f' %self.output_dict['test_time'])
            print('Total time: %0.6f' %self.output_dict['total_time'])
            print('Train accuracy: %0.6f' %self.output_dict['train_accuracy'])
            print('Test accuracy: %0.6f' %self.output_dict['test_accuracy'])
        
            metrics.confusionMatrix(y_hat, y_test, self.output_dict['model'], self.paths)
        
        if self.method == 3:
            
            print('%s fold %i tr_acc %0.3f ts_acc %0.3f tr_time %0.3f ts_time %0.3f dim %i c %s p %s' 
                  %(self.output_dict['model'], self.output_dict['validation'],
                  self.output_dict['train_accuracy'], self.output_dict['test_accuracy'], 
                  self.output_dict['train_time'], self.output_dict['test_time'], 
                  self.output_dict['dimensions'],
                  str(self.output_dict['hype_c']), self.output_dict['hype_penalty']))

class ridge_classifier:
    
    def __init__(self, file, paths, method):
        
        self.file = file
        self.paths = paths
        self.output_dict = {}
        self.method = method
        
    def create_dict(self):
        
        self.output_dict = output.init_dict()
        self.output_dict['model'] = 'Ridge Classifier'
        
    def write(self):
        
        output.write_row(self.file, self.output_dict)
        
    def train(self):
        print('\n--- Running Ridge Classifier ---')
        if self.method == 1:
            print('-- Tuned model on all 50000 samples --')
        if self.method == 2:
            print('-- Tuned model using 10-fold cross validation --')
        if self.method == 3:
            print('-- Grid search using 10-fold cross validation --')
        
        X_train, X_test, y_train, y_test = data.get_data()
        self.create_dict()
        
        if self.method == 1:
            
            self.optimal(X_train, X_test, y_train, y_test)
        
        elif self.method == 2 or self.method == 3:
                
            kth_fold = 1
        
            for train_index, test_index in KFold(n_splits = 10).split(X_train):
                    
                X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
                y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
                
                print('\n-- Cross Validation Fold: %i --' %kth_fold)
                          
                if self.method == 2:
                    
                    self.output_dict['validation'] = kth_fold
                    self.optimal(X_train_cv, X_test_cv, y_train_cv, y_test_cv)
                    
                if self.method == 3:
                    
                    self.gridsearch(X_train_cv, X_test_cv, y_train_cv, y_test_cv)
                             
                kth_fold += 1
                
    def optimal(self, X_train, X_test, y_train, y_test):
        
        self.output_dict['hype_penalty'] = 'l2'
        self.output_dict['hype_c'] = 0.01

        pc = 1000
 
        if pc != X_train.shape[1]:
            start = time()
            svd = PCA(n_components=pc, svd_solver='full')
            X_train = svd.fit_transform(X_train)
            X_test = svd.transform(X_test)
            process_time = time() - start
            
            self.output_dict['preprocess_time'] = process_time
            self.output_dict['retained_variance'] = np.sum(svd.explained_variance_ratio_)
       
        self.predict(X_train, X_test, y_train, y_test)
        
    def gridsearch(self, X_train, X_test, y_train, y_test):
        
        hype_c = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 2, 3, 5]
        components = [10, 20, 30, 50, 100, 200, 500, 1000, 2000, X_train.shape[1]]
        
        for pc in components:
            
            if pc != X_train.shape[1]:
                start = time()
                svd = PCA(n_components=pc, svd_solver='full')
                X_train_r = svd.fit_transform(X_train)
                X_test_r = svd.transform(X_test)                
                self.output_dict['preprocess_time'] = time() - start
                self.output_dict['retained_variance'] = np.sum(svd.explained_variance_ratio_)
            else:
                X_train_r = X_train
                X_test_r = X_test
                self.output_dict['preprocess_time'] = 0
                self.output_dict['retained_variance'] = 1
            
            for c in hype_c:
                
                self.output_dict['hype_c'] = c
                self.predict(X_train_r, X_test_r, y_train, y_test)
        
    def predict(self, X_train, X_test, y_train, y_test):
        
        self.output_dict['dimensions'] = X_train.shape[1]
        self.output_dict['observations'] = X_train.shape[0]
                
        start = time()
        clf = RidgeClassifier(alpha = self.output_dict['hype_c'])
        mdl = clf.fit(X_train, y_train)
        y_hat = mdl.predict(X_train)
        
        self.output_dict['train_time'] = time() - start
        self.output_dict['train_accuracy'] = accuracy_score(y_hat, y_train)
        self.output_dict['train_precision'] = precision_score(y_hat, y_train, average='weighted')
        self.output_dict['train_recall'] = recall_score(y_hat, y_train, average='weighted')
        self.output_dict['train_fscore'] = f1_score(y_hat, y_train, average='weighted')
    
        start = time()
        y_hat = mdl.predict(X_test)

        self.output_dict['test_time'] = time() - start
        self.output_dict['test_accuracy'] = accuracy_score(y_hat, y_test)
        self.output_dict['test_precision'] = precision_score(y_hat, y_test, average='weighted')
        self.output_dict['test_recall'] = recall_score(y_hat, y_test, average='weighted')
        self.output_dict['test_fscore'] = f1_score(y_hat, y_test, average='weighted')
        self.output_dict['total_time'] = self.output_dict['train_time'] + self.output_dict['test_time'] + self.output_dict['preprocess_time']
        
        self.write()
        
        if self.method == 1 or self.method == 2:
            
            print('Time to process: %0.6f' %self.output_dict['preprocess_time'])
            print('Time to train: %0.6f' %self.output_dict['train_time'])
            print('Time to test: %0.6f' %self.output_dict['test_time'])
            print('Total time: %0.6f' %self.output_dict['total_time'])
            print('Train accuracy: %0.6f' %self.output_dict['train_accuracy'])
            print('Test accuracy: %0.6f' %self.output_dict['test_accuracy'])
        
            metrics.confusionMatrix(y_hat, y_test, self.output_dict['model'], self.paths)
        
        if self.method == 3:
            
            print('%s fold %i tr_acc %0.3f ts_acc %0.3f tr_time %0.3f ts_time %0.3f dim %i c %s' 
                  %(self.output_dict['model'], self.output_dict['validation'],
                  self.output_dict['train_accuracy'], self.output_dict['test_accuracy'], 
                  self.output_dict['train_time'], self.output_dict['test_time'], 
                  self.output_dict['dimensions'],
                  str(self.output_dict['hype_c'])))

                       
class sgd_classifier:
    
    def __init__(self, file, paths, method):
        
        self.file = file
        self.paths = paths
        self.output_dict = {}
        self.method = method
        
    def create_dict(self):
        
        self.output_dict = output.init_dict()
        self.output_dict['model'] = 'SGD Classifier'
        
    def write(self):
        
        output.write_row(self.file, self.output_dict)
        
    def train(self):
        print('\n--- Running SGD Classifier ---')
        if self.method == 1:
            print('-- Tuned model on all 50000 samples --')
        if self.method == 2:
            print('-- Tuned model using 10-fold cross validation --')
        if self.method == 3:
            print('-- Grid search using 10-fold cross validation --')
        
        X_train, X_test, y_train, y_test = data.get_data()
        self.create_dict()
        
        if self.method == 1:
            
            self.optimal(X_train, X_test, y_train, y_test)
        
        elif self.method == 2 or self.method == 3:
                
            kth_fold = 1
        
            for train_index, test_index in KFold(n_splits = 10).split(X_train):
                    
                X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
                y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
                
                print('\n-- Cross Validation Fold: %i --' %kth_fold)
                          
                if self.method == 2:
                    
                    self.output_dict['validation'] = kth_fold
                    self.optimal(X_train_cv, X_test_cv, y_train_cv, y_test_cv)
                    
                if self.method == 3:
                    
                    self.gridsearch(X_train_cv, X_test_cv, y_train_cv, y_test_cv)
                             
                kth_fold += 1
                
    def optimal(self, X_train, X_test, y_train, y_test):
        
        self.output_dict['hype_penalty'] = 'l2'
        self.output_dict['hype_c'] = 0.5
        self.output_dict['hype_loss'] = 'modified_huber'

        pc = X_train.shape[1]
 
        if pc != X_train.shape[1]:
            start = time()
            svd = PCA(n_components=pc, svd_solver='full')
            X_train = svd.fit_transform(X_train)
            X_test = svd.transform(X_test)
            process_time = time() - start
            
            self.output_dict['preprocess_time'] = process_time
            self.output_dict['retained_variance'] = np.sum(svd.explained_variance_ratio_)
       
        self.predict(X_train, X_test, y_train, y_test)
        
    def gridsearch(self, X_train, X_test, y_train, y_test):
        
        hype_c = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 2, 3, 5]
        hype_loss = ['hinge', 'squared_hinge', 'modified_huber']
        hype_penalty = ['l1', 'l2', 'elasticnet']
        components = [10, 20, 30, 50, 100, 200, 500, 1000, 2000, X_train.shape[1]]
        
        for pc in components:
            
            if pc != X_train.shape[1]:
                start = time()
                svd = PCA(n_components=pc, svd_solver='full')
                X_train_r = svd.fit_transform(X_train)
                X_test_r = svd.transform(X_test)                
                self.output_dict['preprocess_time'] = time() - start
                self.output_dict['retained_variance'] = np.sum(svd.explained_variance_ratio_)
            else:
                X_train_r = X_train
                X_test_r = X_test
                self.output_dict['preprocess_time'] = 0
                self.output_dict['retained_variance'] = 1
            
            for c in hype_c:
                
                self.output_dict['hype_c'] = c
                                
                for l in hype_loss:
                    
                    self.output_dict['hype_loss'] = l
                                    
                    for p in hype_penalty:
                        
                        self.output_dict['hype_penalty'] = p
                        self.predict(X_train_r, X_test_r, y_train, y_test)
        
    def predict(self, X_train, X_test, y_train, y_test):
        
        self.output_dict['dimensions'] = X_train.shape[1]
        self.output_dict['observations'] = X_train.shape[0]
                
        start = time()
        clf = SGDClassifier(alpha=self.output_dict['hype_c'], loss=self.output_dict['hype_loss'], penalty=self.output_dict['hype_penalty'])
        mdl = clf.fit(X_train, y_train)
        y_hat = mdl.predict(X_train)
        
        self.output_dict['train_time'] = time() - start
        self.output_dict['train_accuracy'] = accuracy_score(y_hat, y_train)
        self.output_dict['train_precision'] = precision_score(y_hat, y_train, average='weighted')
        self.output_dict['train_recall'] = recall_score(y_hat, y_train, average='weighted')
        self.output_dict['train_fscore'] = f1_score(y_hat, y_train, average='weighted')
    
        start = time()
        y_hat = mdl.predict(X_test)

        self.output_dict['test_time'] = time() - start
        self.output_dict['test_accuracy'] = accuracy_score(y_hat, y_test)
        self.output_dict['test_precision'] = precision_score(y_hat, y_test, average='weighted')
        self.output_dict['test_recall'] = recall_score(y_hat, y_test, average='weighted')
        self.output_dict['test_fscore'] = f1_score(y_hat, y_test, average='weighted')
        self.output_dict['total_time'] = self.output_dict['train_time'] + self.output_dict['test_time'] + self.output_dict['preprocess_time']
        
        self.write()
        
        if self.method == 1 or self.method == 2:
            
            print('Time to process: %0.6f' %self.output_dict['preprocess_time'])
            print('Time to train: %0.6f' %self.output_dict['train_time'])
            print('Time to test: %0.6f' %self.output_dict['test_time'])
            print('Total time: %0.6f' %self.output_dict['total_time'])
            print('Train accuracy: %0.6f' %self.output_dict['train_accuracy'])
            print('Test accuracy: %0.6f' %self.output_dict['test_accuracy'])
        
            metrics.confusionMatrix(y_hat, y_test, self.output_dict['model'], self.paths)
        
        if self.method == 3:
            
            print('%s fold %i tr_acc %0.3f ts_acc %0.3f tr_time %0.3f ts_time %0.3f dim %i c %s p %s l %s' 
                  %(self.output_dict['model'], self.output_dict['validation'],
                  self.output_dict['train_accuracy'], self.output_dict['test_accuracy'], 
                  self.output_dict['train_time'], self.output_dict['test_time'], 
                  self.output_dict['dimensions'], str(self.output_dict['hype_c']), 
                  self.output_dict['hype_penalty'], self.output_dict['hype_loss']))

class linear_svc:
    
    def __init__(self, file, paths, method):
        
        self.file = file
        self.method = method
        self.paths = paths
        self.output_dict = {}
        
    def create_dict(self):
        
        self.output_dict = output.init_dict()
        self.output_dict['model'] = 'Linear SVC'
        
    def write(self):
        output.write_row(self.file, self.output_dict)
        
    def train(self):
        print('\n--- Running Linear SVC ---')
        if self.method == 1:
            print('-- Tuned model on all 50000 samples --')
        if self.method == 2:
            print('-- Tuned model using 10-fold cross validation --')
        if self.method == 3:
            print('-- Grid search using 10-fold cross validation --')
        
        X_train, X_test, y_train, y_test = data.get_data()
        self.create_dict()
        
        if self.method == 1:
            
            self.optimal(X_train, X_test, y_train, y_test)
        
        elif self.method == 2 or self.method == 3:
                
            kth_fold = 1
        
            for train_index, test_index in KFold(n_splits = 10).split(X_train):
                    
                X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
                y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
                
                print('\n-- Cross Validation Fold: %i --' %kth_fold)
                          
                if self.method == 2:
                    
                    self.output_dict['validation'] = kth_fold
                    self.optimal(X_train_cv, X_test_cv, y_train_cv, y_test_cv)
                    
                if self.method == 3:
                    
                    self.gridsearch(X_train_cv, X_test_cv, y_train_cv, y_test_cv)
                             
                kth_fold += 1
                         
    def optimal(self, X_train, X_test, y_train, y_test):
                
        self.output_dict['hype_penalty'] = 'l2'
        self.output_dict['hype_loss'] = 'squared_hinge'
        self.output_dict['hype_c'] = 0.01
                  
        pc = 500     
 
        if pc != X_train.shape[1]:
            start = time()
            svd = PCA(n_components=pc, svd_solver='full')
            X_train = svd.fit_transform(X_train)
            X_test = svd.transform(X_test)
            process_time = time() - start
            
            self.output_dict['preprocess_time'] = process_time
            self.output_dict['retained_variance'] = np.sum(svd.explained_variance_ratio_)
       
        self.predict(X_train, X_test, y_train, y_test)
        
    def gridsearch(self, X_train, X_test, y_train, y_test):
        
        hype_c = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 2, 3, 5]
        hype_loss = ['hinge', 'squared_hinge']
        hype_penalty = ['l2']
        components = [10, 20, 30, 50, 100, 200, 500, 1000, 2000, X_train.shape[1]]
        
        for pc in components:
            
            if pc != X_train.shape[1]:
                start = time()
                svd = PCA(n_components=pc, svd_solver='full')
                X_train_r = svd.fit_transform(X_train)
                X_test_r = svd.transform(X_test)
                
                self.output_dict['preprocess_time'] = time() - start
                self.output_dict['retained_variance'] = np.sum(svd.explained_variance_ratio_)
            else:
                X_train_r = X_train
                X_test_r = X_test
                self.output_dict['preprocess_time'] = 0
                self.output_dict['retained_variance'] = 1
        
            for c in hype_c:
                
                self.output_dict['hype_c'] = c
                           
                for l in hype_loss:
                    
                    self.output_dict['hype_loss'] = l
                               
                    for p in hype_penalty:
                        
                        self.output_dict['hype_penalty'] = p
                        self.predict(X_train_r, X_test_r, y_train, y_test)
        
    def predict(self, X_train, X_test, y_train, y_test):
        
        self.output_dict['dimensions'] = X_train.shape[1]
        self.output_dict['observations'] = X_train.shape[0]
    
        start = time()
        clf = LinearSVC(C=self.output_dict['hype_c'], loss=self.output_dict['hype_loss'], penalty=self.output_dict['hype_penalty'])
        mdl = clf.fit(X_train, y_train)
        y_hat = mdl.predict(X_train)
        
        self.output_dict['train_time'] = time() - start
        self.output_dict['train_accuracy'] = accuracy_score(y_hat, y_train)
        self.output_dict['train_precision'] = precision_score(y_hat, y_train, average='weighted')
        self.output_dict['train_recall'] = recall_score(y_hat, y_train, average='weighted')
        self.output_dict['train_fscore'] = f1_score(y_hat, y_train, average='weighted')
    
        start = time()
        y_hat = mdl.predict(X_test)
        
        self.output_dict['test_time'] = time() - start
        self.output_dict['test_accuracy'] = accuracy_score(y_hat, y_test)
        self.output_dict['test_precision'] = precision_score(y_hat, y_test, average='weighted')
        self.output_dict['test_recall'] = recall_score(y_hat, y_test, average='weighted')
        self.output_dict['test_fscore'] = f1_score(y_hat, y_test, average='weighted')
        self.output_dict['total_time'] = self.output_dict['train_time'] + self.output_dict['test_time'] + self.output_dict['preprocess_time']
        
        self.write()
        
        if self.method == 1 or self.method == 2:
            
            print('Time to process: %0.6f' %self.output_dict['preprocess_time'])
            print('Time to train: %0.6f' %self.output_dict['train_time'])
            print('Time to test: %0.6f' %self.output_dict['test_time'])
            print('Total time: %0.6f' %self.output_dict['total_time'])
            print('Train accuracy: %0.6f' %self.output_dict['train_accuracy'])
            print('Test accuracy: %0.6f' %self.output_dict['test_accuracy'])
        
            metrics.confusionMatrix(y_hat, y_test, self.output_dict['model'], self.paths)
        
        if self.method == 3:    
        
            print('%s fold %i tr_acc %0.3f ts_acc %0.3f tr_time %0.3f ts_time %0.3f dim %i c %s p %s l %s' 
                  %(self.output_dict['model'], self.output_dict['validation'],
                  self.output_dict['train_accuracy'], self.output_dict['test_accuracy'], 
                  self.output_dict['train_time'], self.output_dict['test_time'], 
                  self.output_dict['dimensions'], str(self.output_dict['hype_c']), 
                  self.output_dict['hype_penalty'], self.output_dict['hype_loss']))

class nearest_neighbours:
    
    def __init__(self, file, paths, method):
        self.file = file
        self.method = method
        self.paths = paths
        self.output_dict = {}
        
    def create_dict(self):
        
        self.output_dict = output.init_dict()
        self.output_dict['model'] = 'Nearest Neighbours'
        
    def write(self):
        
        output.write_row(self.file, self.output_dict)
        
    def train(self):
        print('\n--- Running Nearest Neighbours ---')
        if self.method == 1:
            print('-- Tuned model on all 50000 samples --')
        if self.method == 2:
            print('-- Tuned model using 10-fold cross validation --')
        if self.method == 3:
            print('-- Grid search using 10-fold cross validation --')
        
        X_train, X_test, y_train, y_test = data.get_data()
        self.create_dict()
        
        if self.method == 1:
            
            self.optimal(X_train, X_test, y_train, y_test)
        
        elif self.method == 2 or self.method == 3:
                
            kth_fold = 1
        
            for train_index, test_index in KFold(n_splits = 10).split(X_train):
                    
                X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
                y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
                
                print('\n-- Cross Validation Fold: %i --' %kth_fold)
                          
                if self.method == 2:
                    
                    self.output_dict['validation'] = kth_fold
                    self.optimal(X_train_cv, X_test_cv, y_train_cv, y_test_cv)
                    
                if self.method == 3:
                    
                    self.gridsearch(X_train_cv, X_test_cv, y_train_cv, y_test_cv)
                             
                kth_fold += 1
                         
    def optimal(self, X_train, X_test, y_train, y_test):
        
        self.output_dict['hype_k'] = 20

        pc = 20        
 
        if pc != X_train.shape[1]:
            start = time()
            svd = PCA(n_components=pc, svd_solver='full')
            X_train = svd.fit_transform(X_train)
            X_test = svd.transform(X_test)
            process_time = time() - start
            
            self.output_dict['preprocess_time'] = process_time
            self.output_dict['retained_variance'] = np.sum(svd.explained_variance_ratio_)
       
        self.predict(X_train, X_test, y_train, y_test)
        
    def gridsearch(self, X_train, X_test, y_train, y_test):
        
        hype_k = [5, 10, 20, 30, 50, 100]
        components = [10, 20, 30, 50, 100, 200, 500, 1000, 2000, X_train.shape[1]]
        
        for pc in components:
            
            if pc != X_train.shape[1]:
                start = time()
                svd = PCA(n_components=pc, svd_solver='full')
                X_train_r = svd.fit_transform(X_train)
                X_test_r = svd.transform(X_test)
                
                self.output_dict['preprocess_time'] = time() - start
                self.output_dict['retained_variance'] = np.sum(svd.explained_variance_ratio_)
            else:
                X_train_r = X_train
                X_test_r = X_test
                self.output_dict['preprocess_time'] = 0
                self.output_dict['retained_variance'] = 1
        
            for k in hype_k:
                
                self.output_dict['hype_k'] = k
                
                if k <= X_train.shape[0]:
                    
                   self.predict(X_train_r, X_test_r, y_train, y_test)
        
    
    def predict(self, X_train, X_test, y_train, y_test):
        
        self.output_dict['dimensions'] = X_train.shape[1]
        self.output_dict['observations'] = X_train.shape[0]
        
        start = time()
        clf = KNeighborsClassifier(n_neighbors=self.output_dict['hype_k'])
        mdl = clf.fit(X_train, y_train)
        y_hat = mdl.predict(X_train)
        
        self.output_dict['train_time'] = time() - start
        self.output_dict['train_accuracy'] = accuracy_score(y_hat, y_train)
        self.output_dict['train_precision'] = precision_score(y_hat, y_train, average='weighted')
        self.output_dict['train_recall'] = recall_score(y_hat, y_train, average='weighted')
        self.output_dict['train_fscore'] = f1_score(y_hat, y_train, average='weighted')
    
        start = time()
        y_hat = mdl.predict(X_test)
        
        self.output_dict['test_time'] = time() - start
        self.output_dict['test_accuracy'] = accuracy_score(y_hat, y_test)
        self.output_dict['test_precision'] = precision_score(y_hat, y_test, average='weighted')
        self.output_dict['test_recall'] = recall_score(y_hat, y_test, average='weighted')
        self.output_dict['test_fscore'] = f1_score(y_hat, y_test, average='weighted')
        self.output_dict['total_time'] = self.output_dict['train_time'] + self.output_dict['test_time'] + self.output_dict['preprocess_time']
    
        self.write()
        
        if self.method == 1 or self.method == 2:
            
            print('Time to process: %0.6f' %self.output_dict['preprocess_time'])
            print('Time to train: %0.6f' %self.output_dict['train_time'])
            print('Time to test: %0.6f' %self.output_dict['test_time'])
            print('Total time: %0.6f' %self.output_dict['total_time'])
            print('Train accuracy: %0.6f' %self.output_dict['train_accuracy'])
            print('Test accuracy: %0.6f' %self.output_dict['test_accuracy'])
        
            metrics.confusionMatrix(y_hat, y_test, self.output_dict['model'], self.paths)
        
        if self.method == 3:
            
            print('%s fold %i tr_acc %0.3f ts_acc %0.3f tr_time %0.3f ts_time %0.3f dim %i k %i' 
                  %(self.output_dict['model'], self.output_dict['validation'],
                  self.output_dict['train_accuracy'], self.output_dict['test_accuracy'], 
                  self.output_dict['train_time'], self.output_dict['test_time'], 
                  self.output_dict['dimensions'], self.output_dict['hype_k'], 
                  ))

class random_forest:
    
    def __init__(self, file, paths, method):
        self.file = file
        self.method = method
        self.paths = paths
        self.output_dict = {}
        
    def create_dict(self):
        
        self.output_dict = output.init_dict()
        self.output_dict['model'] = 'Random Forest'
        
    def write(self):
        
        output.write_row(self.file, self.output_dict)
        
    def train(self):
        print('\n--- Running Random Forest ---')
        if self.method == 1:
            print('-- Tuned model on all 50000 samples --')
        if self.method == 2:
            print('-- Tuned model using 10-fold cross validation --')
        if self.method == 3:
            print('-- Grid search using 10-fold cross validation --')
        
        X_train, X_test, y_train, y_test = data.get_data()
        self.create_dict()
        
        if self.method == 1:
            
            self.optimal(X_train, X_test, y_train, y_test)
        
        elif self.method == 2 or self.method == 3:
                
            kth_fold = 1
        
            for train_index, test_index in KFold(n_splits = 10).split(X_train):
                    
                X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
                y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
                
                print('\n-- Cross Validation Fold: %i --' %kth_fold)
                          
                if self.method == 2:
                    
                    self.output_dict['validation'] = kth_fold
                    self.optimal(X_train_cv, X_test_cv, y_train_cv, y_test_cv)
                    
                if self.method == 3:
                    
                    self.gridsearch(X_train_cv, X_test_cv, y_train_cv, y_test_cv)
                             
                kth_fold += 1
                         
    def optimal(self, X_train, X_test, y_train, y_test):
        
        self.output_dict['hype_e'] = 200
        self.output_dict['hype_k'] = 0.25
        self.output_dict['hype_d'] = 10
                                   
        pc = 100   
 
        if pc != X_train.shape[1]:
            start = time()
            svd = PCA(n_components=pc, svd_solver='full', whiten = True)
            X_train = svd.fit_transform(X_train)
            X_test = svd.transform(X_test)
            process_time = time() - start
            
            self.output_dict['preprocess_time'] = process_time
            self.output_dict['retained_variance'] = np.sum(svd.explained_variance_ratio_)
       
        self.predict(X_train, X_test, y_train, y_test)
        
    def gridsearch(self, X_train, X_test, y_train, y_test):

        hype_d = [2, 4, 6, 8, 10]        
        hype_e = [5, 10, 20, 50, 100, 200]
        hype_k = [0.05, 0.25, 0.5, 0.75, 1]
        components = [10, 20, 30, 50, 100, 200, 500, 1000, 2000, X_train.shape[1]]
        
        for pc in components:
            
            if pc != X_train.shape[1]:
                start = time()
                svd = PCA(n_components=pc, svd_solver='full')
                X_train_r = svd.fit_transform(X_train)
                X_test_r = svd.transform(X_test)                
                self.output_dict['preprocess_time'] = time() - start
                self.output_dict['retained_variance'] = np.sum(svd.explained_variance_ratio_)
            else:
                X_train_r = X_train
                X_test_r = X_test
                self.output_dict['preprocess_time'] = 0
                self.output_dict['retained_variance'] = 1

            for e in hype_e:
                
                self.output_dict['hype_e'] = e
                           
                for k in hype_k:
                    
                    self.output_dict['hype_k'] = k
                               
                    for d in hype_d:
                        
                        self.output_dict['hype_d'] = d
                        self.predict(X_train_r, X_test_r, y_train, y_test)
        
    def predict(self, X_train, X_test, y_train, y_test):
        
        self.output_dict['dimensions'] = X_train.shape[1]
        self.output_dict['observations'] = X_train.shape[0]
                     
        start = time()
        clf = RandomForestClassifier(n_estimators=self.output_dict['hype_e'], max_features=self.output_dict['hype_k'], max_depth=self.output_dict['hype_d'])
        mdl = clf.fit(X_train, y_train)
        y_hat = mdl.predict(X_train)
        
        self.output_dict['train_time'] = time() - start
        self.output_dict['train_accuracy'] = accuracy_score(y_hat, y_train)
        self.output_dict['train_precision'] = precision_score(y_hat, y_train, average='weighted')
        self.output_dict['train_recall'] = recall_score(y_hat, y_train, average='weighted')
        self.output_dict['train_fscore'] = f1_score(y_hat, y_train, average='weighted')
    
        start = time()
        y_hat = mdl.predict(X_test)
        
        self.output_dict['test_time'] = time() - start
        self.output_dict['test_accuracy'] = accuracy_score(y_hat, y_test)
        self.output_dict['test_precision'] = precision_score(y_hat, y_test, average='weighted')
        self.output_dict['test_recall'] = recall_score(y_hat, y_test, average='weighted')
        self.output_dict['test_fscore'] = f1_score(y_hat, y_test, average='weighted')
        self.output_dict['total_time'] = self.output_dict['train_time'] + self.output_dict['test_time'] + self.output_dict['preprocess_time']
        
        self.write()
        
        if self.method == 1 or self.method == 2:
            
            print('Time to process: %0.6f' %self.output_dict['preprocess_time'])
            print('Time to train: %0.6f' %self.output_dict['train_time'])
            print('Time to test: %0.6f' %self.output_dict['test_time'])
            print('Total time: %0.6f' %self.output_dict['total_time'])
            print('Train accuracy: %0.6f' %self.output_dict['train_accuracy'])
            print('Test accuracy: %0.6f' %self.output_dict['test_accuracy'])
        
            metrics.confusionMatrix(y_hat, y_test, self.output_dict['model'], self.paths)
        
        if self.method == 3:
                    
            print('%s fold %i tr_acc %0.3f ts_acc %0.3f tr_time %0.3f ts_time %0.3f dim %i e %i k %0.1f d %i' 
                  %(self.output_dict['model'], self.output_dict['validation'],
                  self.output_dict['train_accuracy'], self.output_dict['test_accuracy'], 
                  self.output_dict['train_time'], self.output_dict['test_time'], 
                  self.output_dict['dimensions'], self.output_dict['hype_e'], 
                  self.output_dict['hype_k'], self.output_dict['hype_d']))

class extra_trees:
    
    def __init__(self, file, paths, method):
        self.file = file
        self.method = method
        self.paths = paths
        self.output_dict = {}
        
    def create_dict(self):
        
        self.output_dict = output.init_dict()
        self.output_dict['model'] = 'Extra Trees'
        
    def write(self):
        
        output.write_row(self.file, self.output_dict)
        
    def train(self):
        print('\n--- Running Extra Trees ---')
        if self.method == 1:
            print('-- Tuned model on all 50000 samples --')
        if self.method == 2:
            print('-- Tuned model using 10-fold cross validation --')
        if self.method == 3:
            print('-- Grid search using 10-fold cross validation --')
        
        X_train, X_test, y_train, y_test = data.get_data()
        self.create_dict()
        
        if self.method == 1:
            
            self.optimal(X_train, X_test, y_train, y_test)
        
        elif self.method == 2 or self.method == 3:
                
            kth_fold = 1
        
            for train_index, test_index in KFold(n_splits = 10).split(X_train):
                    
                X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
                y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
                
                print('\n-- Cross Validation Fold: %i --' %kth_fold)
                          
                if self.method == 2:
                    
                    self.output_dict['validation'] = kth_fold
                    self.optimal(X_train_cv, X_test_cv, y_train_cv, y_test_cv)
                    
                if self.method == 3:
                    
                    self.gridsearch(X_train_cv, X_test_cv, y_train_cv, y_test_cv)
                             
                kth_fold += 1
                         
    def optimal(self, X_train, X_test, y_train, y_test):
        
        self.output_dict['hype_e'] = 200
        self.output_dict['hype_k'] = 0.25
        self.output_dict['hype_d'] = 10
                                   
        pc = 100   
 
        if pc != X_train.shape[1]:
            start = time()
            svd = PCA(n_components=pc, svd_solver='full', whiten = True)
            X_train = svd.fit_transform(X_train)
            X_test = svd.transform(X_test)
            process_time = time() - start
            
            self.output_dict['preprocess_time'] = process_time
            self.output_dict['retained_variance'] = np.sum(svd.explained_variance_ratio_)
       
        self.predict(X_train, X_test, y_train, y_test)
        
    def gridsearch(self, X_train, X_test, y_train, y_test):

        hype_d = [2, 4, 6, 8, 10]        
        hype_e = [5, 10, 20, 50, 100, 200]
        hype_k = [0.05, 0.25, 0.5, 0.75, 1]
        components = [10, 20, 30, 50, 100, 200, 500, 1000, 2000, X_train.shape[1]]
        
        for pc in components:
            
            if pc != X_train.shape[1]:
                start = time()
                svd = PCA(n_components=pc, svd_solver='full')
                X_train_r = svd.fit_transform(X_train)
                X_test_r = svd.transform(X_test)                
                self.output_dict['preprocess_time'] = time() - start
                self.output_dict['retained_variance'] = np.sum(svd.explained_variance_ratio_)
            else:
                X_train_r = X_train
                X_test_r = X_test
                self.output_dict['preprocess_time'] = 0
                self.output_dict['retained_variance'] = 1

            for e in hype_e:
                
                self.output_dict['hype_e'] = e
                           
                for k in hype_k:
                    
                    self.output_dict['hype_k'] = k
                               
                    for d in hype_d:
                        
                        self.output_dict['hype_d'] = d
                        self.predict(X_train_r, X_test_r, y_train, y_test)
        
    def predict(self, X_train, X_test, y_train, y_test):
        
        self.output_dict['dimensions'] = X_train.shape[1]
        self.output_dict['observations'] = X_train.shape[0]
                     
        start = time()
        clf = ExtraTreesClassifier(n_estimators=self.output_dict['hype_e'], max_features=self.output_dict['hype_k'], max_depth=self.output_dict['hype_d'])
        mdl = clf.fit(X_train, y_train)
        y_hat = mdl.predict(X_train)
        
        self.output_dict['train_time'] = time() - start
        self.output_dict['train_accuracy'] = accuracy_score(y_hat, y_train)
        self.output_dict['train_precision'] = precision_score(y_hat, y_train, average='weighted')
        self.output_dict['train_recall'] = recall_score(y_hat, y_train, average='weighted')
        self.output_dict['train_fscore'] = f1_score(y_hat, y_train, average='weighted')
    
        start = time()
        y_hat = mdl.predict(X_test)
        
        self.output_dict['test_time'] = time() - start
        self.output_dict['test_accuracy'] = accuracy_score(y_hat, y_test)
        self.output_dict['test_precision'] = precision_score(y_hat, y_test, average='weighted')
        self.output_dict['test_recall'] = recall_score(y_hat, y_test, average='weighted')
        self.output_dict['test_fscore'] = f1_score(y_hat, y_test, average='weighted')
        self.output_dict['total_time'] = self.output_dict['train_time'] + self.output_dict['test_time'] + self.output_dict['preprocess_time']
        
        self.write()
        
        if self.method == 1 or self.method == 2:
            
            print('Time to process: %0.6f' %self.output_dict['preprocess_time'])
            print('Time to train: %0.6f' %self.output_dict['train_time'])
            print('Time to test: %0.6f' %self.output_dict['test_time'])
            print('Total time: %0.6f' %self.output_dict['total_time'])
            print('Train accuracy: %0.6f' %self.output_dict['train_accuracy'])
            print('Test accuracy: %0.6f' %self.output_dict['test_accuracy'])
        
            metrics.confusionMatrix(y_hat, y_test, self.output_dict['model'], self.paths)
        
        if self.method == 3:
                    
            print('%s fold %i tr_acc %0.3f ts_acc %0.3f tr_time %0.3f ts_time %0.3f dim %i e %i k %0.1f d %i' 
                  %(self.output_dict['model'], self.output_dict['validation'],
                  self.output_dict['train_accuracy'], self.output_dict['test_accuracy'], 
                  self.output_dict['train_time'], self.output_dict['test_time'], 
                  self.output_dict['dimensions'], self.output_dict['hype_e'], 
                  self.output_dict['hype_k'], self.output_dict['hype_d']))   

class gradient_boost:
    
    def __init__(self, file, paths, method):
        self.file = file
        self.method = method
        self.paths = paths
        self.output_dict = {}
        
    def create_dict(self):
        
        self.output_dict = output.init_dict()
        self.output_dict['model'] = 'Gradient Boosting Machine'
        
    def write(self):
        
        output.write_row(self.file, self.output_dict)
        
    def train(self):
        print('\n--- Running Gradient Boosting Machine ---')
        if self.method == 1:
            print('-- Tuned model on all 50000 samples --')
        if self.method == 2:
            print('-- Tuned model using 10-fold cross validation --')
        if self.method == 3:
            print('-- Grid search using 10-fold cross validation --')
        
        X_train, X_test, y_train, y_test = data.get_data()
        self.create_dict()
        
        if self.method == 1:
            
            self.optimal(X_train, X_test, y_train, y_test)
        
        elif self.method == 2 or self.method == 3:
                
            kth_fold = 1
        
            for train_index, test_index in KFold(n_splits = 10).split(X_train):
                    
                X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
                y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
                
                print('\n-- Cross Validation Fold: %i --' %kth_fold)
                          
                if self.method == 2:
                    
                    self.output_dict['validation'] = kth_fold
                    self.optimal(X_train_cv, X_test_cv, y_train_cv, y_test_cv)
                    
                if self.method == 3:
                    
                    self.gridsearch(X_train_cv, X_test_cv, y_train_cv, y_test_cv)
                             
                kth_fold += 1
                         
    def optimal(self, X_train, X_test, y_train, y_test):
        
        self.output_dict['hype_e'] = 200
        self.output_dict['hype_k'] = 0.25
        self.output_dict['hype_d'] = 10
                                   
        pc = 100   
 
        if pc != X_train.shape[1]:
            start = time()
            svd = PCA(n_components=pc, svd_solver='full', whiten = True)
            X_train = svd.fit_transform(X_train)
            X_test = svd.transform(X_test)
            process_time = time() - start
            
            self.output_dict['preprocess_time'] = process_time
            self.output_dict['retained_variance'] = np.sum(svd.explained_variance_ratio_)
       
        self.predict(X_train, X_test, y_train, y_test)
        
    def gridsearch(self, X_train, X_test, y_train, y_test):

        hype_d = [2, 4, 6, 8, 10]        
        hype_e = [5, 10, 20, 50, 100, 200]
        hype_k = [0.05, 0.25, 0.5, 0.75, 1]
        components = [10, 20, 30, 50, 100, 200, 500, 1000, 2000, X_train.shape[1]]
        
        for pc in components:
            
            if pc != X_train.shape[1]:
                start = time()
                svd = PCA(n_components=pc, svd_solver='full')
                X_train_r = svd.fit_transform(X_train)
                X_test_r = svd.transform(X_test)                
                self.output_dict['preprocess_time'] = time() - start
                self.output_dict['retained_variance'] = np.sum(svd.explained_variance_ratio_)
            else:
                X_train_r = X_train
                X_test_r = X_test
                self.output_dict['preprocess_time'] = 0
                self.output_dict['retained_variance'] = 1

            for e in hype_e:
                
                self.output_dict['hype_e'] = e
                           
                for k in hype_k:
                    
                    self.output_dict['hype_k'] = k
                               
                    for d in hype_d:
                        
                        self.output_dict['hype_d'] = d
                        self.predict(X_train_r, X_test_r, y_train, y_test)
        
    def predict(self, X_train, X_test, y_train, y_test):
        
        self.output_dict['dimensions'] = X_train.shape[1]
        self.output_dict['observations'] = X_train.shape[0]
                     
        start = time()
        clf = GradientBoostingClassifier(n_estimators=self.output_dict['hype_e'], max_features=self.output_dict['hype_k'], max_depth=self.output_dict['hype_d'])
        mdl = clf.fit(X_train, y_train)
        y_hat = mdl.predict(X_train)
        
        self.output_dict['train_time'] = time() - start
        self.output_dict['train_accuracy'] = accuracy_score(y_hat, y_train)
        self.output_dict['train_precision'] = precision_score(y_hat, y_train, average='weighted')
        self.output_dict['train_recall'] = recall_score(y_hat, y_train, average='weighted')
        self.output_dict['train_fscore'] = f1_score(y_hat, y_train, average='weighted')
    
        start = time()
        y_hat = mdl.predict(X_test)
        
        self.output_dict['test_time'] = time() - start
        self.output_dict['test_accuracy'] = accuracy_score(y_hat, y_test)
        self.output_dict['test_precision'] = precision_score(y_hat, y_test, average='weighted')
        self.output_dict['test_recall'] = recall_score(y_hat, y_test, average='weighted')
        self.output_dict['test_fscore'] = f1_score(y_hat, y_test, average='weighted')
        self.output_dict['total_time'] = self.output_dict['train_time'] + self.output_dict['test_time'] + self.output_dict['preprocess_time']
        
        self.write()
        
        if self.method == 1 or self.method == 2:
            
            print('Time to process: %0.6f' %self.output_dict['preprocess_time'])
            print('Time to train: %0.6f' %self.output_dict['train_time'])
            print('Time to test: %0.6f' %self.output_dict['test_time'])
            print('Total time: %0.6f' %self.output_dict['total_time'])
            print('Train accuracy: %0.6f' %self.output_dict['train_accuracy'])
            print('Test accuracy: %0.6f' %self.output_dict['test_accuracy'])
        
            metrics.confusionMatrix(y_hat, y_test, self.output_dict['model'], self.paths)
        
        if self.method == 3:
                    
            print('%s fold %i tr_acc %0.3f ts_acc %0.3f tr_time %0.3f ts_time %0.3f dim %i e %i k %0.1f d %i' 
                  %(self.output_dict['model'], self.output_dict['validation'],
                  self.output_dict['train_accuracy'], self.output_dict['test_accuracy'], 
                  self.output_dict['train_time'], self.output_dict['test_time'], 
                  self.output_dict['dimensions'], self.output_dict['hype_e'], 
                  self.output_dict['hype_k'], self.output_dict['hype_d']))
class ada_boost:
    
    def __init__(self, file, paths, method):
        self.file = file
        self.method = method
        self.paths = paths
        self.output_dict = {}
        
    def create_dict(self):
        
        self.output_dict = output.init_dict()
        self.output_dict['model'] = 'Adaboost'
        
    def write(self):
        
        output.write_row(self.file, self.output_dict)
        
    def train(self):
        print('\n--- Running Adaboost ---')
        if self.method == 1:
            print('-- Tuned model on all 50000 samples --')
        if self.method == 2:
            print('-- Tuned model using 10-fold cross validation --')
        if self.method == 3:
            print('-- Grid search using 10-fold cross validation --')
        
        X_train, X_test, y_train, y_test = data.get_data()
        self.create_dict()
        
        if self.method == 1:
            
            self.optimal(X_train, X_test, y_train, y_test)
        
        elif self.method == 2 or self.method == 3:
                
            kth_fold = 1
        
            for train_index, test_index in KFold(n_splits = 10).split(X_train):
                    
                X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
                y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
                
                print('\n-- Cross Validation Fold: %i --' %kth_fold)
                          
                if self.method == 2:
                    
                    self.output_dict['validation'] = kth_fold
                    self.optimal(X_train_cv, X_test_cv, y_train_cv, y_test_cv)
                    
                if self.method == 3:
                    
                    self.gridsearch(X_train_cv, X_test_cv, y_train_cv, y_test_cv)
                             
                kth_fold += 1
                         
    def optimal(self, X_train, X_test, y_train, y_test):
        
        self.output_dict['hype_e'] = 200
                                   
        pc = 100   
 
        if pc != X_train.shape[1]:
            start = time()
            svd = PCA(n_components=pc, svd_solver='full', whiten = True)
            X_train = svd.fit_transform(X_train)
            X_test = svd.transform(X_test)
            process_time = time() - start
            
            self.output_dict['preprocess_time'] = process_time
            self.output_dict['retained_variance'] = np.sum(svd.explained_variance_ratio_)
       
        self.predict(X_train, X_test, y_train, y_test)
        
    def gridsearch(self, X_train, X_test, y_train, y_test):
       
        hype_e = [5, 10, 20, 50, 100, 200]
        components = [10, 20, 30, 50, 100, 200, 500, 1000, 2000, X_train.shape[1]]
        
        for pc in components:
            
            if pc != X_train.shape[1]:
                start = time()
                svd = PCA(n_components=pc, svd_solver='full')
                X_train_r = svd.fit_transform(X_train)
                X_test_r = svd.transform(X_test)                
                self.output_dict['preprocess_time'] = time() - start
                self.output_dict['retained_variance'] = np.sum(svd.explained_variance_ratio_)
            else:
                X_train_r = X_train
                X_test_r = X_test
                self.output_dict['preprocess_time'] = 0
                self.output_dict['retained_variance'] = 1

            for e in hype_e:
                
                self.output_dict['hype_e'] = e
                self.predict(X_train_r, X_test_r, y_train, y_test)
        
    def predict(self, X_train, X_test, y_train, y_test):
        
        self.output_dict['dimensions'] = X_train.shape[1]
        self.output_dict['observations'] = X_train.shape[0]
                     
        start = time()
        clf = AdaBoostClassifier(n_estimators=self.output_dict['hype_e'])
        mdl = clf.fit(X_train, y_train)
        y_hat = mdl.predict(X_train)
        
        self.output_dict['train_time'] = time() - start
        self.output_dict['train_accuracy'] = accuracy_score(y_hat, y_train)
        self.output_dict['train_precision'] = precision_score(y_hat, y_train, average='weighted')
        self.output_dict['train_recall'] = recall_score(y_hat, y_train, average='weighted')
        self.output_dict['train_fscore'] = f1_score(y_hat, y_train, average='weighted')
    
        start = time()
        y_hat = mdl.predict(X_test)
        
        self.output_dict['test_time'] = time() - start
        self.output_dict['test_accuracy'] = accuracy_score(y_hat, y_test)
        self.output_dict['test_precision'] = precision_score(y_hat, y_test, average='weighted')
        self.output_dict['test_recall'] = recall_score(y_hat, y_test, average='weighted')
        self.output_dict['test_fscore'] = f1_score(y_hat, y_test, average='weighted')
        self.output_dict['total_time'] = self.output_dict['train_time'] + self.output_dict['test_time'] + self.output_dict['preprocess_time']
        
        self.write()
        
        if self.method == 1 or self.method == 2:
            
            print('Time to process: %0.6f' %self.output_dict['preprocess_time'])
            print('Time to train: %0.6f' %self.output_dict['train_time'])
            print('Time to test: %0.6f' %self.output_dict['test_time'])
            print('Total time: %0.6f' %self.output_dict['total_time'])
            print('Train accuracy: %0.6f' %self.output_dict['train_accuracy'])
            print('Test accuracy: %0.6f' %self.output_dict['test_accuracy'])
        
            metrics.confusionMatrix(y_hat, y_test, self.output_dict['model'], self.paths)
        
        if self.method == 3:
                    
            print('%s fold %i tr_acc %0.3f ts_acc %0.3f tr_time %0.3f ts_time %0.3f dim %i e %i' 
                  %(self.output_dict['model'], self.output_dict['validation'],
                  self.output_dict['train_accuracy'], self.output_dict['test_accuracy'], 
                  self.output_dict['train_time'], self.output_dict['test_time'], 
                  self.output_dict['dimensions'], self.output_dict['hype_e'], 
                  ))