import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier


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

def run_experiment_classifier(X, y, splits, summary_class=True, verbose = False):
    summary = []
    current_train_index = []

    for train_index, test_index, train_interval, test_interval in splits:
        if verbose:
            print(test_index)
        X_test, Y_test = X.loc[test_index], y[test_index]

        if not np.array_equal(train_index, current_train_index):
            X_train, Y_train = X.loc[train_index], y[train_index]
            if verbose:
                print("shapes: " + str((X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)))
                print(Y_train.value_counts())
            
            regressor = RandomForestClassifier(random_state=0,n_estimators=100,n_jobs=-1)
            regressor.fit(X_train, Y_train)
            current_train_index = train_index
            
        test_predict = regressor.predict(X_test)

        report = classification_report(Y_test, test_predict, output_dict = True)
        auc = roc_auc_score(Y_test, regressor.predict_proba(X_test)[:,regressor.classes_.tolist().index(summary_class)])


        if verbose:
            print(train_interval, test_interval)
            print(Y_test.value_counts())
            print("test set (precision / recall / f-score / auc): ", report['True']['precision'], report['True']['recall'], report['True']['f1-score'], auc)

        summary = np.append(summary, [report[str(summary_class)]['precision'], report[str(summary_class)]['recall'], report[str(summary_class)]['f1-score'], auc, train_interval, test_interval, Y_train.shape[0], Y_test.shape[0], Y_test.value_counts()[summary_class]], axis=0)
        
    return pd.DataFrame(np.reshape(summary, [-1,9]), columns=['precision', 'recall', 'f1-score', 'auc', 'train', 'test', 'train_size', 'test_size', 'label_size'])

def run_experiment_classifier_y(X, y, splits, window_size, step_size, summary_class=True, return_index = True, verbose = False):
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

class VotingPretrained:
    def __init__(self, clf_list, weights):
        self.clf_list = clf_list
        self.weights = weights
        self.le_ = LabelEncoder()
        
    def fit(self, X, y):
        self.le_.fit(y)
        return
    
    def predict_proba(self, X):
        pred = np.asarray([clf.predict_proba(X) for clf in self.clf_list])
        predict_proba = np.average(pred, axis=0, weights=self.weights)
        return predict_proba

    def predict(self, X):
        return self.le_.inverse_transform(np.argmax(self.predict_proba(X), axis=1))


def run_experiment_classifier_voting(X, y, splits, summary_class=True, verbose=False, threshold=0.5):
    summary = []
    current_train_index = []
    regressors = []
    last_test = -1

    for train_index, test_index, train_interval, test_interval in splits:
        X_test, Y_test = X.loc[test_index], y[test_index]

        if not np.array_equal(train_index, current_train_index):
            X_train, Y_train = X.loc[train_index], y[train_index]
            if verbose:
                print("shapes: " + str((X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)))
                print(Y_train.value_counts())
            
            regressor = RandomForestClassifier(random_state=0,n_estimators=100,n_jobs=-1)
            regressor.fit(X_train, Y_train)
            current_train_index = train_index

            # Esto corre el riesgo de descartar un clasificador que no es malo si el conjunto de tests que le toca es especialmente complicado
            if 0 < last_test < threshold:
                regressors.pop()

            regressors.append(regressor)
            
            classifier = VotingPretrained(regressors, compute_weights(len(regressors)))
            classifier.fit(X_train, Y_train)
            last_test = -1
            #classifier = VotingClassifier(regressors, voting='soft', weights=compute_weights(len(regressors)))
            #classifier.fit(X_train, Y_train)

            
        test_predict = classifier.predict(X_test)

        report = classification_report(Y_test, test_predict, output_dict = True)
        auc = roc_auc_score(Y_test, classifier.predict_proba(X_test)[:,regressor.classes_.tolist().index(summary_class)])

        if last_test == -1:
            test_regressor = regressor.predict(X_test)
            report_reg = classification_report(Y_test, test_regressor, output_dict=True)
            last_test = report_reg[str(summary_class)]['f1-score']
            
        if verbose:
            print(train_interval, test_interval)
            print(Y_test.value_counts())
        #print("test set (precision / recall / f-score / auc): ", report['True']['precision'], report['True']['recall'], report['True']['f1-score'], auc)

        summary = np.append(summary, [report[str(summary_class)]['precision'], report[str(summary_class)]['recall'], report[str(summary_class)]['f1-score'], auc, train_interval, test_interval, Y_train.shape[0], Y_test.shape[0], Y_test.value_counts()[summary_class]], axis=0)
        
    return pd.DataFrame(np.reshape(summary, [-1,9]), columns=['precision', 'recall', 'f1-score', 'auc', 'train', 'test', 'train_size', 'test_size', 'label_size'])        
