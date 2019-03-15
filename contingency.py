import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
    
def compute_y_chi2(y, splits, verbose=False):
    summary = []
    
    for train_index, test_index, train_interval, test_interval in splits:
        data = pd.concat([pd.DataFrame({'d': y[train_index], 'i': str(train_interval)}),
                          pd.DataFrame({'d': y[test_index], 'i': str(test_interval)})], axis=0)
        
        contingency = pd.crosstab(data.i, data.d)
        if verbose:
            print(str(train_interval), str(test_interval))
            display(contingency)

        if contingency.size == 0:
            continue

        result = chi2_contingency(contingency)
        
        summary = np.append(summary, [train_interval, test_interval, result[0], result[1]], axis=0)

    return pd.DataFrame(np.reshape(summary, [-1,4]), columns=['train_interval', 'test_interval', 'chi2', 'pvalue'])

def compute_all_chi2(X, splits, verbose=False):
    summary = []
    detail = []
    for train_index, test_index, train_interval, test_interval in splits:
        pvalues = []
        for c in X.columns:
            data = pd.concat([pd.DataFrame({'d': X[c][train_index], 'i': str(train_interval)}),
                              pd.DataFrame({'d': X[c][test_index], 'i': str(test_interval)})], axis=0)

            contingency = pd.crosstab(data.i, data.d)    
            if contingency.size == 0:
                continue

            result = chi2_contingency(contingency)
            pvalues.append(result[1] > 0.05)
            if verbose:
                print(str(train_interval), str(test_interval))
                display(contingency)
                print(result[1])
            detail = np.append(detail, [train_interval, test_interval, c, result[1]], axis=0)

        count_pvalues = sum(pvalues)
        summary = np.append(summary, [train_interval, test_interval, count_pvalues], axis=0)

    return pd.DataFrame(np.reshape(summary, [-1,3]), columns=['train_interval', 'test_interval', 'num_of_h0']), pd.DataFrame(np.reshape(detail, [-1,4]), columns=['train_interval', 'test_interval', 'attrib', 'pvalue'])