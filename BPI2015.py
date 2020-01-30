#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'predictive-monitoring-evolution'))
	print(os.getcwd())
except:
	pass

#%%
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 60)
bpi2015= []
for c in range(5):
    bpi2015.append(pd.read_csv("bpi2015_"+str(c+1)+".csv", low_memory=False))
    print(bpi2015[c].shape)
    print(bpi2015[c].columns)
    

#%% [markdown]
# # Preprocessing
# Starts the preprocessing to remove null values in the 5 datasets

#%%
for c in bpi2015:
    print(np.all(c.startTime == c.completeTime))


#%%
pd.concat([c.isnull().sum() for c in bpi2015], axis=1)


#%%
bpi2015[0][['termName', 'caseProcedure', 'Responsible_actor', 'caseStatus', 'Includes_subCases', 'parts', 'landRegisterID']].dtypes


#%%
for c in bpi2015:
    c.loc[:,['termName', 'caseProcedure','caseStatus', 'Includes_subCases', 'parts']] = c.loc[:,['termName', 'caseProcedure','caseStatus', 'Includes_subCases', 'parts']].fillna('NA')


#%%
print(np.any(pd.concat([c['Responsible_actor'].value_counts() for c in bpi2015], axis=1) == 0))
print(np.any(pd.concat([c['landRegisterID'].value_counts() for c in bpi2015], axis=1) == 0))


#%%
for c in bpi2015:
    c.loc[:,['Responsible_actor', 'landRegisterID']] = c.loc[:,['Responsible_actor', 'landRegisterID']].fillna(0)


#%%
print(np.any(pd.concat([c['SUMleges'].value_counts() for c in bpi2015], axis=1) == 0))


#%%
for c in bpi2015:
    c.loc[:,['SUMleges']] = c.loc[:,['SUMleges']].fillna(0)


#%%
for c in bpi2015:
    c['HasConceptCase'] = ~c['IDofConceptCase'].isnull()


#%%
pd.concat([c.isnull().sum() for c in bpi2015], axis=1)


#%%
pd.concat([c.nunique() for c in bpi2015], axis=1)

#%% [markdown]
# # Encoding the datasets

#%%
import EventLog as el
logs = [el.EventLog(c, 'case', 'completeTime') for c in bpi2015]


#%%
from sklearn.preprocessing import OneHotEncoder

encoder = el.LogEncoder(transformers = [('static_drop', 'drop', ['case_type', 'startDate', 'endDate', 'endDatePlanned', 'last_phase', 'IDofConceptCase']),
                                     ('static_keep', 'keep', ['requestComplete', 'HasConceptCase']),
                                     ('static_onehot', el.WrapperEncoder(logs[0].id_column,OneHotEncoder(sparse=False)), ['termName', 'caseProcedure', 'Responsible_actor', 'caseStatus', 'Includes_subCases', 'parts', 'landRegisterID']),
                                     ('dynamic_drop', 'drop', ['action_code', 'activityNameNL', 'planned', 'dateStop', 'dateFinished', 'dueDate', 'question']),
                                     ('dynamic_keep', 'keep', ['SUMleges']),
                                     ('dynamic_freq', el.FrequencyEncoder(logs[0].id_column), ['event', 'org:resource', 'activityNameEN','monitoringResource']),
                                     ('timestamp', el.TimestampFeatures(logs[0].id_column, ['event_order', 'time_from_start', 'elapsed_time_from_event']), [logs[0].timestamp_column])])


encoder.check_unused(logs[0])


#%%
datasets = [encoder.fit_transform(l) for l in logs]


#%%
[(dataset.isnull().sum() > 0).sum() for dataset in datasets]

#%% [markdown]
# Here we compute the target for the prediction

#%%
y = []
for log in logs:
    mask_scr = log.df['activityNameEN'] == 'send confirmation receipt'
    mask_rmd = log.df['activityNameEN'] == 'retrieve missing data'
    event_order = log.df.groupby(log.id_column).cumcount()
    rmd = pd.concat([event_order[mask_rmd], log.df[log.id_column]], axis=1).groupby('case').transform('max')
    scr = pd.concat([event_order[mask_scr], log.df[log.id_column]], axis=1).groupby('case').transform('max')
    y.append(~(scr.isnull() | (rmd > scr))[0])
    
X = [el.transform_timedeltas(dataset) for dataset in datasets]


#%%
[logs[i].df.loc[y[i], logs[i].id_column].nunique() for i in range(0,5)]

#%% [markdown]
# We first make sure that the shapes of X and y look nice

#%%
[(X[i].shape, y[i].shape) for i in range(5)]

#%% [markdown]
# # Experiments

#%%
from splitters import TimeCaseSplit

def print_split(splitter, strategy):
    for i in range(5):
        print('Dataset '+str(i))
        prev_ti = None
        count_train = 0
        count_test = 0
        for tri,tsi,ti,ts in splitter.split(X[i],y[i],logs[i].df[logs[i].id_column], logs[i].df[logs[i].timestamp_column], strategy):
            if prev_ti != ti:
                print('Train ' + str(count_train) + ' has '+ str(count_test))
                print('--Train interval '+ str(ti) + "["+str(len(tri))+"]")
                prev_ti = ti
                count_test = 0
                count_train = count_train + 1
            print('    ' + str(ts)+ "["+str(len(tsi))+"]")
            
            count_test = count_test + 1


#%%
from splitters import CummulativeStrategy, NonCummulativeStrategy, SamplingStrategy, DriftStrategy
from experiments import compute_weights

#%%
split = TimeCaseSplit(train_size=pd.DateOffset(months=6), train_freq=pd.DateOffset(months=6), test_freq=pd.DateOffset(months=6), test_periods=50, threshold=60)
print_split(split, strategy=NonCummulativeStrategy(train_size=pd.DateOffset(months=6)))
print_split(split, strategy=SamplingStrategy(train_size=61, weights=compute_weights(5)))
print_split(split, strategy=DriftStrategy(drifts=[pd.Timestamp('2013-06-06T06:06')], threshold=61))

#%%
from splitters import NumberCaseSplit
split = NumberCaseSplit(train_size=100, train_step=50, test_freq=100, test_periods=50, threshold=60)
print_split(split, strategy=NonCummulativeStrategy(train_size=100))
print("-----")
print_split(split, strategy=SamplingStrategy(train_size=100, weights=compute_weights(5)))
print("-----")
print_split(split, strategy=DriftStrategy(drifts=[pd.Timestamp('2013-06-06T06:06')], threshold=61))

#%% [markdown]
# ## RQ1: Does the dataset change over time?

#%%
catcols = set(X[0].columns.values) - set(['SUMleges','event_order_completeTime', 'time_from_start_completeTime', 'elapsed_time_from_event_completeTime'])


#%%
from splitters import TimeCaseSplit
from contingency import compute_all_chi2

allchi2 = []
for i in range(5):
    tcs = TimeCaseSplit(train_size=pd.DateOffset(months=6), train_freq=pd.DateOffset(months=6), test_freq=pd.DateOffset(months=6), test_periods=50, threshold=60)
    allchi2.append(compute_all_chi2(X[i].loc[:,catcols], tcs.split(X[i],y[i],logs[i].df[logs[i].id_column], logs[i].df[logs[i].timestamp_column], strategy=NonCummulativeStrategy(train_size=pd.DateOffset(months=6)))))


#%%
import seaborn as sns
import matplotlib.pyplot as plt
def draw_heatmap(data):
    s = len(data.columns)+1
    yticklabels=['[I'+str(i)+"] "+data.columns[i-1].left.strftime('%m-%Y') + " to " + data.columns[i-1].right.strftime('%m-%Y') for i in range(1,s)]
    xticklabels=['[I'+str(i)+']' for i in range(1,s)]
    fig, ax = plt.subplots()
    ax = sns.heatmap(data, yticklabels = yticklabels, xticklabels=xticklabels)
    ax.set_xlabel("")
    ax.set_ylabel("")    
    
    return ax


#%%
[draw_heatmap(allchi2[i][0].pivot(columns='test_interval', index='train_interval', values='num_of_h0').fillna(0)) for i in range(5)]


#%%
for i in range(5):
    allchi2[i][0].to_csv('allchi2_'+str(i)+'.csv')

#%% [markdown]
# # RQ2: Does time have an influence on the quality of the models?
# To answer this question, we are going to analyse the evolution of model performance using the first two strategies (new models each time and updating a model that includes all information):

#%%
from experiments import run_experiment, shape_summary, VotingPretrainedClassifier
from splitters import NumberCaseSplit
from sklearn.ensemble import RandomForestClassifier

# Utilities for experiments

def splits(i, splitter, strategy):
    return splitter.split(X[i],
                          y[i],
                          logs[i].df[logs[i].id_column], 
                          logs[i].df[logs[i].timestamp_column], 
                          strategy=strategy)


def launch_experiment(months_size, months_freq, months_test, datasets=range(0,5)):
    tcs = TimeCaseSplit(train_size=pd.DateOffset(months=months_size), 
                        train_freq=pd.DateOffset(months=months_freq), 
                        test_freq=pd.DateOffset(months=months_test), 
                        test_periods=14, 
                        threshold=100)
    summary_X = [None] * 5
    summary_X_V = [None] * 5
    summary_X_F = [None] * 5
    clf = RandomForestClassifier(random_state=0,n_estimators=100,n_jobs=-1)
    for i in datasets:
        agg_clf = VotingPretrainedClassifier(weights=compute_weights)
        summary_X[i]= run_experiment(X[i], ~y[i], splits(i, tcs, CummulativeStrategy()), clf)
        summary_X_F[i] = run_experiment(X[i], ~y[i], splits(i, tcs, NonCummulativeStrategy(train_size=pd.DateOffset(months=months_size))), clf)
        summary_X_V[i] = run_experiment(X[i], ~y[i], splits(i, tcs, NonCummulativeStrategy(train_size=pd.DateOffset(months=months_size))), clf, aggregate_clf=agg_clf, verbose=True)
        
    return summary_X, summary_X_V, summary_X_F

def launch_experiment_rolling(size, freq, window, steps, datasets=range(0,5)):
    ncs = NumberCaseSplit(train_size=size,
                          train_step=freq)
    summary_X = [None] * 5
    summary_X_F = [None] * 5
    clf = RandomForestClassifier(random_state=0,n_estimators=100,n_jobs=-1)
    for i in datasets:
        summary_X[i]= run_experiment(X[i], ~y[i], 
                                                splits(i, ncs, CummulativeStrategy()),
                                                clf=clf,
                                                report=RollingReport(window_size=window, 
                                                                     step_size=steps, 
                                                                     summary_class=False), 
                                                verbose=True)
        summary_X_F[i]= run_experiment(X[i], ~y[i], 
                                                splits(i, ncs, NonCummulativeStrategy(train_size=size)),
                                                clf=clf,
                                                report=RollingReport(window_size=window, 
                                                                     step_size=steps, 
                                                                     summary_class=False), 
                                                verbose=True)
        
    return summary_X, summary_X_F

def launch_experiment_number(size, freq, test, datasets=range(0,5)):
    ncs = NumberCaseSplit(train_size=size, train_step=freq, test_freq=test, test_periods=20, threshold=100)

    summary_X = [None] * 5
    summary_X_V = [None] * 5
    summary_X_F = [None] * 5
    summary_X_S = [None] * 5
    clf = RandomForestClassifier(random_state=0,n_estimators=100,n_jobs=-1)

    for i in datasets:
        summary_X[i]= run_experiment(X[i], ~y[i], splits(i, ncs, CummulativeStrategy()), clf=clf, verbose=True)
        summary_X_F[i] = run_experiment(X[i], ~y[i], splits(i, ncs, NonCummulativeStrategy(train_size=size)), clf=clf)
        agg_clf = VotingPretrainedClassifier(weights=compute_weights)
        summary_X_V[i] = run_experiment(X[i], ~y[i], splits(i, ncs, NonCummulativeStrategy(train_size=size)), clf=clf, aggregate_clf=agg_clf)
        summary_X_S[i] = run_experiment(X[i], ~y[i], splits(i, ncs, SamplingStrategy(train_size=size, weights=compute_weights(5))), clf=clf)

    return summary_X, summary_X_V, summary_X_F, summary_X_S




#%%
summary_999 = launch_experiment(9,9,9)


#%%
summary_666 = launch_experiment(6,6,6)


#%%
summary_300150150 = launch_experiment_number(300, 150, 150)

#%% [markdown]
# These are the heatmaps for strategy non-cummulative for each of the 5 datasets

#%%
[draw_heatmap(shape_summary(summary_300150150[0][i]).xs('f1-score', axis=1, drop_level=True).fillna(0)) for i in range(5)]

#%% [markdown]
# And these are the heatmaps for strategy cummulative for the 5 datasets

#%%
[draw_heatmap(shape_summary(summary_300150150[1][i]).xs('f1-score', axis=1, drop_level=True).fillna(0)) for i in range(5)]

#%% [markdown]
# # RQ3: How does the different update strategies compare against each other?
#%% [markdown]
# Finally, we compute execute the experiment for all types of strategies. The final value is the mean of the F-Scores in the diagonal of the matrix obtained.

#%%
def extract_values(summary, step = 0):
    mask = summary[0].sort_values(['train','test']).groupby('train').cumcount() <= step
    return pd.concat([summary[i][mask]['f1-score'] for i in range(len(summary))],axis=1)


#%%
summary_300_fscore = [None]*5
summary_300_fscore = [extract_values([summary_300150150[i][j] for i in range(4)]) for j in range(5)]

for i in range(5):
    summary_300_fscore[i].columns = ['X','V','F','S']


#%%
pd.concat([summary_300_fscore[i].mean() for i in range(5)], axis=1)

#%% [markdown]
# Here, we use the T-test related to check whether there is a significant difference between each column. The result is that only the difference between 'V' and 'X is statistically significant.

#%%
from scipy import stats
[(d,i,j,stats.ttest_rel(summary_300_fscore[d][i], summary_300_fscore[d][j])) for d in range(5) for ii,i in enumerate(summary_300_fscore[d].columns) for jj,j in enumerate(summary_300_fscore[d].columns) if i < j]




# %%
