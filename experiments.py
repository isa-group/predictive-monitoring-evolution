import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone

def compare_m(summ1, summ2, datasets=range(5), reindex=False, verbose=False):
    diffs = []
    for i in datasets:
        diffs.append(compare(summ1, summ2, reindex, verbose))

    return diffs

def compare(summ1, summ2, datasets=range(5), reindex=False, verbose=False):
    if not reindex:
        diff = shape_summary(summ1[i]) - shape_summary(summ2[i])
    else:
        diff = shape_summary(summ1[i]).reset_index(drop=True) - shape_summary(summ2[i]).reset_index(drop=True)

    if verbose:
        display(diff)
        print(diff.mean().mean())

    return diff

def compare_diag_m(summ1, summ2, datasets=range(5)):
    result = []
    for i in datasets:
        result.append(compare_diag(summ1, summ2))        
    
    return result

def compare_diag_step(summ1, summ2, step=1):
    step1 = summ1.shape[1] + step
    diag1 = summ1.flatten()[::step1]
    step2 = summ2.shape[2] + step
    diag2 = summ2.flatten()[::step2]
    diff = diag1 - diag2

    return np.mean(diff)

def compare_diag(summ1, summ2):
    diag1 = np.diag(shape_summary(summ1))
    diag2 = np.diag(shape_summary(summ2))    
    diff = diag1[0:min(diag1.shape[0], diag2.shape[0])] - diag2[0:min(diag1.shape[0], diag2.shape[0])]

    return np.mean(diff)


def shape_summary(summary, values=['f1-score']):
    return summary.pivot(columns='test', index='train', values=values) 


def run_experiment_classifier_old(X, y, splits, window_size, step_size, summary_class=True, return_index = True, verbose = False):
    summary = []
    current_train_index = []
    summary_index = []

    for train_index, test_index, train_interval, test_interval in splits:
        if verbose:
            print(test_index)
            
        X_test, Y_test = X.loc[test_index], y[test_index]
        X_train, Y_train = X.loc[train_index], y[train_index]

        regressor = RandomForestClassifier(random_state=0,n_estimators=100,n_jobs=-1)
        regressor.fit(X_train, Y_train)

        test_predict = regressor.predict(X_test)

        report = classification_report(Y_test, test_predict, output_dict = True)
        auc = roc_auc_score(Y_test, regressor.predict_proba(X_test)[:,regressor.classes_.tolist().index(summary_class)])

        if verbose:
            display(test_predict)
            display(Y_test)
            
        indexer = np.arange(window_size)[None, :] + step_size*np.arange(len(test_predict)//step_size)[:,None]
        indexer = np.where(indexer >= len(test_predict), len(test_predict)-1, indexer)
        test_predict_indexer = test_predict[indexer]
        Y_test_indexer = Y_test.values[indexer]
        rolling_f1 = [classification_report(Y_test_indexer[i],test_predict_indexer[i], output_dict=True)[str(summary_class)]['f1-score'] for i in range(indexer.shape[0])]

        if verbose:
            print(train_interval, test_interval)
            print(Y_test.value_counts())
            print("test set (precision / recall / f-score / auc): ", report['True']['precision'], report['True']['recall'], report['True']['f1-score'], auc)

        summary.append(rolling_f1)

        if return_index:
            summary_index.append(Y_test.index.values[indexer])

    if return_index:
        return summary, summary_index
    else:
        return summary


def compute_weights(length):
    w = [math.exp(-x/3) for x in range(length,0,-1)]
    return [float(i)/sum(w) for i in w]

class VotingPretrainedClassifier:    
    '''Special classifier that can ensemble previously trained 
    classifiers that can be added using method append. 

    It allows to give different weights to each classifier by 
    means of weights, which is a function that given a number
    of elements, return the weight of each element. It is 
    designed to give different weights to the classifiers that
    were trained with older information.

    It also allows to remove classifiers whose performance
    (measured as f1-score) is lower than a predetermined 
    threshold by means of method clean_last.

    Parameters
    ----------
    weights : function
        Function that given a number of elements, returns 
        a list with the weights of each element.
    summary_class : boolean, default = True
        Target class that is used to remove classifier
        with bad performance.
    clf_list : list of classifiers, default = None
        Initial list of classifiers to use in the
        ensemble. By default, the list is empty
    threshold : integer, default = -1
        Threshold to remove a classifier with bad 
        performance. No classifier is removed if 
        threshold < 0.
    '''
    def __init__(self, weights, summary_class=True, clf_list=None, threshold=-1):
        if clf_list is None:
            self.clf_list = []
        else:
            self.clf_list = clf_list

        self.threshold = threshold
        self.weights = weights
        self.summary_class = summary_class
        self.le_ = LabelEncoder()
        self.classes_ = []
    
    def append(self, clf):
        self.clf_list.append(clf)

    def clean_last(self, X_test, Y_test):
        if self.threshold > 0 and len(self.clf_list) > 1:
            last = self.clf_list[-1]
            test_result = last.predict(X_test)
            report_reg = classification_report(Y_test, test_result, output_dict=True)
            last_test = report_reg[str(self.summary_class)]['f1-score']

            if last_test < threshold:
                self.clf_list.pop()
        
        return

    def fit(self, X, y):
        self.le_.fit(y)
        self.classes_ = self.le_.classes_
        return
    
    def predict_proba(self, X):
        pred = np.asarray([clf.predict_proba(X) for clf in self.clf_list])
        weights = self.weights(len(self.clf_list))
        predict_proba = np.average(pred, axis=0, weights=weights)
        return predict_proba

    def predict(self, X):
        return self.le_.inverse_transform(np.argmax(self.predict_proba(X), axis=1))

class AggregatedReport:
    '''Standard report used to evaluate the performance of a predictive 
    monitoring model in a cross validation setting.

    For each test, it returns the following information: 'precision', 
    'recall', 'f1-score', 'auc', 'train interval', 'test interval', 
    'train shape', 'test shape', 'number of samples of the summary_class'

    Parameters
    ----------
    summary_class : any, default = True
        The class for which the metric is computed
    '''
    def __init__(self, summary_class=True):
        super().__init__()

        self.summary = []
        self.summary_class = summary_class

    def add(self, clf, X_test, Y_test, Y_train, train_interval, test_interval):
        test_predict = clf.predict(X_test)

        report = classification_report(Y_test, test_predict, output_dict = True)
        auc = roc_auc_score(Y_test, clf.predict_proba(X_test)[:,clf.classes_.tolist().index(self.summary_class)])
        self.summary = np.append(self.summary, 
                                 [report[str(self.summary_class)]['precision'], 
                                  report[str(self.summary_class)]['recall'], 
                                  report[str(self.summary_class)]['f1-score'], 
                                  auc, 
                                  train_interval, 
                                  test_interval, 
                                  Y_train.shape[0], 
                                  Y_test.shape[0], 
                                  Y_test.value_counts()[self.summary_class]], axis=0)

    def report(self):
        return pd.DataFrame(np.reshape(self.summary, [-1,9]), 
                            columns=['precision', 'recall', 'f1-score', 'auc', 
                                     'train', 'test', 'train_size', 'test_size', 
                                     'label_size'])        
    

class RollingReport:
    '''Report used to obtain a rolling evaluation metric of the performance 
    of a predictive monitoring model.

    Parameters
    ----------
    window_size : int
        The size of the window used in the rolling metric
    step_size : int
        The size of each step of the window used in the rolling metric so that
        it doesn't have to be recomputed for each new sample
    summary_class : any, default = True
        The class for which the metric is computed
    metric : string, default='f1-score'
        The metric that is computed. It has to be one of the metrics provided
        by classification_report
    return_index : bool, default=True
        If true, the index of y samples of each window is returned together 
        with the result of the metric
    '''
    def __init__(self, window_size, step_size, summary_class=True, metric='f1-score', return_index=True):
        super().__init__()

        self.summary = []
        self.summary_index = []
        self.window_size = window_size
        self.step_size = step_size
        self.summary_class = summary_class
        self.metric = metric
        self.return_index = return_index

    def add(self, clf, X_test, Y_test, Y_train, train_interval, test_interval):
        test_predict = clf.predict(X_test)

        indexer = np.arange(self.window_size)[None, :] + self.step_size*np.arange(len(test_predict)//self.step_size)[:,None]
        indexer = np.where(indexer >= len(test_predict), len(test_predict)-1, indexer)
        test_predict_indexer = test_predict[indexer]
        Y_test_indexer = Y_test.values[indexer]
        rolling_f1 = [classification_report(Y_test_indexer[i],test_predict_indexer[i], output_dict=True)[str(self.summary_class)][self.metric] for i in range(indexer.shape[0])]

        self.summary.append(rolling_f1)

        if self.return_index:
            self.summary_index.append(Y_test.index.values[indexer])

    def report(self):
        if self.return_index:
            return self.summary, self.summary_index
        else:
            return self.summary


def run_experiment(X, y, splits, clf, report=None, aggregate_clf=None, verbose=False):
    """Executes an experiments for the dataset provided by X and y, and using 
    the splits and classifier provided. It also supports the possibility of
    using a mechanism to ensemble classifiers from different splits.

    The report of experiments are stored by the class given in report.

    Parameters
    ----------

    X : (n_samples, n_features)
        The features
    y : (n_samples,)
        The target class
    splits : (n_splits,)
        An iterable over the splits in the dataset. It should provide
        a train_index, test_index, train_interval, test_interval, where
        the intervals are the dates of each split
    clf : Classifier
        A classifier that must implement the methods fit and predict. It 
        is not used directly, but cloned for each split, so it must also
        be clonable.
    report : Report, default = AggregatedReport
        The type of report that is used as a result of the experiment.
    aggregate_clf : Classifier, default = None
        A special type of classifier that can be used to ensemble the
        classifiers obtained from previous iterations.
    verbose : boolean, default = False
        If true, it prints additional information about the iterations that
        are being executed. It is useful to debug the process.
    """

    if report is None:
        report = AggregatedReport()

    current_train_index = []
    last_X_test, last_Y_test = None, None

    for train_index, test_index, train_interval, test_interval in splits:
        X_test, Y_test = X.loc[test_index], y[test_index]

        if not np.array_equal(train_index, current_train_index):
            X_train, Y_train = X.loc[train_index], y[train_index]
            if verbose:
                print("shapes: " + str((X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)))
                print(Y_train.value_counts())
            
            classifier = clone(clf)
            classifier.fit(X_train, Y_train)
            current_train_index = train_index

            if not aggregate_clf is None:
                if not last_X_test is None:
                    # This might discard a classifier that is not bad if its test set is particularly difficult
                    aggregate_clf.clean_last(last_X_test, last_Y_test)

                aggregate_clf.append(classifier)
                aggregate_clf.fit(X_train, Y_train)
                #classifier = VotingClassifier(regressors, voting='soft', weights=compute_weights(len(regressors)))
                #classifier.fit(X_train, Y_train)
                classifier = aggregate_clf


        report.add(classifier, X_test, Y_test, Y_train, train_interval, test_interval)

        last_X_test, last_Y_test = X_test, Y_test

        if verbose:
            print(train_interval, test_interval)
            print(Y_test.value_counts())
        #print("test set (precision / recall / f-score / auc): ", report['True']['precision'], report['True']['recall'], report['True']['f1-score'], auc)

        
    return report.report()        


