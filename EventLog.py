import pandas as pd
import numpy as np
from time import time
import sys
from sklearn.base import TransformerMixin
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils import safe_indexing
from sklearn.preprocessing import OneHotEncoder
from itertools import chain
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier


class EventLog:
    def __init__(self, df, id_column='id', timestamp_column='timestamp', timestamp_format="%Y-%m-%dT%H:%M:%S%z"):
        self.id_column = id_column
        self.timestamp_column = timestamp_column
        self.df = df.sort_values([self.id_column, self.timestamp_column]).reset_index(drop=True)
        
        if not pd.core.dtypes.common.is_datetime_or_timedelta_dtype(self.df[self.timestamp_column]):
            temp = pd.to_datetime(self.df[self.timestamp_column], infer_datetime_format=True,utc=True, errors='raise')
            print('Found ' + str(np.any(pd.isnull(temp))))
            self.df[self.timestamp_column] = temp
        
    def static_columns(self):
        return [col for col in self.df.columns.values if ((col not in [self.id_column, self.timestamp_column]) and (not np.any(self.change_in_case_mask(col))))]
    
    def mask_case(self, mask, mask_type='any'):
        if mask_type in ['any', 'all']:
            func = np.any if mask_type == 'any' else np.all
            return pd.concat([mask, self.df[self.id_column]], keys=['mask', 'id'], axis=1).groupby('id').transform(func)['mask']
        elif mask_type == 'to_end':
            return pd.concat([mask, self.df[self.id_column]], keys=['mask', 'id'], axis=1).groupby('id')['mask'].cummax()
        else:
            raise ValueError('mask_type not valid')
            
    def start_case_mask(self):
        return (self.df[self.id_column] != self.df[self.id_column].shift(1))
    
    def end_case_mask(self):
        return (self.df[self.id_column] != self.df[self.id_column].shift(-1))
        
    def ids_for(self, mask):
        return self.df.loc[mask, self.id_column]
    
    def change_in_case_mask(self, column):
        return (self.df[column] != self.df[column].shift()) & (self.df[self.id_column] == self.df[self.id_column].shift())
        
    def transform_case_aware(self, func, update_first = None, update_last = None):
        col = func(self.df)
        if not update_first is None:
            col[self.df[self.id_column] != self.df[self.id_column].shift()] = update_first
        if not update_last is None:
            col[self.df[self.id_column] != self.df[self.id_column].shift(-1)] = update_last
        return col  
    
    def plot_events_by_date(self):
        return self.df.groupby(self.df[self.timestamp_column].dt.date)[self.id_column].count().plot()
    
    def plot_cases_by_date(self, when='end'):
        op = 'max' if when == 'end' else 'min'
        time = self.df.groupby(self.id_column)[self.timestamp_column].transform(op)
            
        return self.df.groupby(time.dt.date)[self.id_column].nunique().plot()
        
class LogEncoder:
    def __init__(self, transformers):
        self.transformers = transformers
        
    def fit(self, log):
        for name, encoder, columns in self.transformers:
            print('Fitting ' + name)
            if encoder not in ['keep', 'drop']:
                select = columns + [log.id_column]
                encoder.fit(log.df[select])
        return self
    
    def check_unused(self, log):
        used = [c for _,_,columns in self.transformers for c in columns]
        return [c for c in log.df.columns if c not in used]
        
    def transform(self, log):
        X_res = []
        for name, encoder, columns in self.transformers:
            print('Transforming '+ name)
            if encoder == 'drop':
                continue
            elif encoder == 'keep':
                X_res.append(log.df[columns])
            else: 
                X_res.append(encoder.transform(log.df[columns + [log.id_column]]))
            
        return pd.concat(X_res, axis=1)
    
    def fit_transform(self, log):
        return self.fit(log).transform(log)
            
class WrapperEncoder:
    def __init__(self, id_column, encoder):
        self.encoder = encoder
        self.id_column = id_column
    
    def fit(self, X, y = None):
        self.encoder.fit(X.drop(self.id_column, axis=1))
        return self
        
    def transform(self, X):
        result = self.encoder.transform(X.drop(self.id_column, axis=1))
        return pd.DataFrame(data=result, columns=self.encoder.get_feature_names())
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class TimestampFeatures:
    def __init__(self, id_column, features):
        self.features = features
        self.id_column = id_column
        
    def fit(self, X, y = None):
        return self

    def _transform_case_aware(self, X, func, update_first = None, update_last = None):
        col = func(X)
        if not update_first is None:
            shifted = X[self.id_column].shift()
            col[(X[self.id_column] != shifted) | (shifted.isnull())] = update_first
        if not update_last is None:
            shifted = X[self.id_column].shift(-1)
            col[(X[self.id_column] != shifted) | (shifted.isnull())] = update_last
        return col      
        
    def transform(self, X):
        X_data = []
        X_cols = []
        for column in X.columns.values:
            if column != self.id_column:
                if 'event_order' in self.features:
                    X_data.append(X.groupby(self.id_column).cumcount())
                    X_cols.append('event_order_'+column)
                if 'last_time' in self.features:
                    X_data.append(X.groupby(self.id_column)[column].transform('max'))
                    X_cols.append('last_time_'+column)
                if 'time_from_start' in self.features:
                    X_data.append(X[column] - X.groupby(self.id_column)[column].transform('min'))
                    X_cols.append('time_from_start_'+column)
                if 'remaining_time' in self.features:
                    X_data.append(X.groupby(self.id_column)[column].transform('max') - X[column])
                    X_cols.append('remaining_time_'+column)
                if 'elapsed_time_from_event' in self.features:
                    X_data.append(self._transform_case_aware(X, func=lambda x: x[column].diff(), update_first=pd.Timedelta(0)))
                    X_cols.append('elapsed_time_from_event_'+column)
        
        result = pd.concat(X_data, axis=1)
        result.columns = X_cols
        
        return result

    def fit_transform(self, log):
        return self.fit(log).transform(log)
    
class FrequencyEncoder(OneHotEncoder):
    def __init__(self, id_column, handle_unknown='error'):
        self.handle_unknown = handle_unknown
        self.id_column = id_column
        self._categories = 'auto'
        
    def fit(self,X, y = None):
        self._fit(X.drop(self.id_column, axis=1), handle_unknown=self.handle_unknown)
        return self      
    
    def transform(self, X):
        X_copy = X.copy()
        X_res = []
        for column in X.columns.values:
            if column != self.id_column:
                X_copy[column+'_count'] = X_copy.groupby([self.id_column, column]).cumcount()+1
                Xi = X_copy.set_index(column, append=True)[column+'_count'].unstack(level=1, fill_value=0)
                Xi.columns = [column + '_' + str(x) for x in Xi.columns]
                X_res.append(Xi)
        return pd.concat(X_res, axis=1)
    
    def fit_transform(self, X, y= None):
        return self.fit(X).transform(X)
                

class WindowEncoder():
    def __init__(self, id_column, window):
        self.id_column = id_column
        self.window = window
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):        
        X_res = pd.concat([X.groupby('id').shift(i) for i in range(self.window - 1, 0, -1)], axis=1)
        X_res.columns = [col+'_'+str(i) for i in range(self.window-1,0,-1) for col in X.columns.values if col != self.id_column]
        return pd.concat([X_res, X], axis=1).drop(self.id_column, axis=1)
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

def transform_timedeltas(X):
    if type(X) == pd.core.series.Series:
        return X.dt.seconds
    else:
        timedeltas = X.select_dtypes('timedelta').columns.values
        df_X = X.drop(timedeltas, axis=1)
        for c in timedeltas:
            df_X[c] = X[c].dt.seconds

        return df_X
            

class AggregateTransformer(TransformerMixin):
    
    def __init__(self, case_id_col, cat_cols, num_cols, boolean=False, fillna=True):
        self.case_id_col = case_id_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        
        self.boolean = boolean
        self.fillna = fillna
        
        self.columns = None
        
        self.fit_time = 0
        self.transform_time = 0
    
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        start = time()
        
        # transform numeric cols
        if len(self.num_cols) > 0:
            dt_numeric = X.groupby(self.case_id_col)[self.num_cols].agg({'mean':np.mean, 'max':np.max, 'min':np.min, 'sum':np.sum, 'std':np.std})
            dt_numeric.columns = ['_'.join(col).strip() for col in dt_numeric.columns.values]
            
        # transform cat cols
        dt_transformed = pd.get_dummies(X[self.cat_cols])
        dt_transformed[self.case_id_col] = X[self.case_id_col]
        del X
        if self.boolean:
            dt_transformed = dt_transformed.groupby(self.case_id_col).max()
        else:
            dt_transformed = dt_transformed.groupby(self.case_id_col).sum()
        
        # concatenate
        if len(self.num_cols) > 0:
            dt_transformed = pd.concat([dt_transformed, dt_numeric], axis=1)
            del dt_numeric
        
        # fill missing values with 0-s
        if self.fillna:
            dt_transformed = dt_transformed.fillna(0)
            
        # add missing columns if necessary
        if self.columns is None:
            self.columns = dt_transformed.columns
        else:
            missing_cols = [col for col in self.columns if col not in dt_transformed.columns]
            for col in missing_cols:
                dt_transformed[col] = 0
            dt_transformed = dt_transformed[self.columns]
        
        self.transform_time = time() - start
        return dt_transformed
    
    def get_feature_names(self):
        return self.columns


class AggregateTransformer(TransformerMixin):
    
    def __init__(self, case_id_col, cat_cols, num_cols, boolean=False, fillna=True):
        self.case_id_col = case_id_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        
        self.boolean = boolean
        self.fillna = fillna
        
        self.columns = None
        
        self.fit_time = 0
        self.transform_time = 0
    
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        start = time()
        
        # transform numeric cols
        if len(self.num_cols) > 0:
            dt_numeric = X.groupby(self.case_id_col)[self.num_cols].agg({'mean':np.mean, 'max':np.max, 'min':np.min, 'sum':np.sum, 'std':np.std})
            dt_numeric.columns = ['_'.join(col).strip() for col in dt_numeric.columns.values]
            
        # transform cat cols
        dt_transformed = pd.get_dummies(X[self.cat_cols])
        dt_transformed[self.case_id_col] = X[self.case_id_col]
        del X
        if self.boolean:
            dt_transformed = dt_transformed.groupby(self.case_id_col).max()
        else:
            dt_transformed = dt_transformed.groupby(self.case_id_col).sum()
        
        # concatenate
        if len(self.num_cols) > 0:
            dt_transformed = pd.concat([dt_transformed, dt_numeric], axis=1)
            del dt_numeric
        
        # fill missing values with 0-s
        if self.fillna:
            dt_transformed = dt_transformed.fillna(0)
            
        # add missing columns if necessary
        if self.columns is None:
            self.columns = dt_transformed.columns
        else:
            missing_cols = [col for col in self.columns if col not in dt_transformed.columns]
            for col in missing_cols:
                dt_transformed[col] = 0
            dt_transformed = dt_transformed[self.columns]
        
        self.transform_time = time() - start
        return dt_transformed
    
    def get_feature_names(self):
        return self.columns            

def run_experiment(X, y, splits):
    summary = []
    current_train_index = []

    for train_index, test_index, train_interval, test_interval in splits:
        X_test, Y_test = X.loc[test_index], y[test_index]

        if not np.array_equal(train_index, current_train_index):
            X_train, Y_train = X.loc[train_index], y[train_index]
            print("shapes: " + str((X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)))
            
            regressor = RandomForestRegressor(random_state=0,n_jobs=-1)
            regressor.fit(X_train, Y_train)
            current_train_index = train_index
            
        test_predict = regressor.predict(X_test)

        test_mae = mean_absolute_error(Y_test, test_predict)
        test_mse = mean_squared_error(Y_test, test_predict)

        print(train_interval, test_interval)
        print("test set (MAE / MSE): ", test_mae, test_mse)

        summary = np.append(summary, [test_mae, test_mse, train_interval, test_interval], axis=0)

    return summary    



class LogExperiments:
    def __init__(self, log):
        self.log = log
        self.df = log.df
        self.id_column = log.id_column
        self.timestamp_column = log.timestamp_column
    

    
    def run_all_tests(self, splitter, target = 'remaining_time', drop=[], case_length=-1):
        """Just a facility to run the default configuration of tests easier"""
        
        df = self.df.sort_values([self.id_column, self.timestamp_column])
        
        timedeltas = df.select_dtypes('timedelta').columns.values
        df_X = df.drop(np.concatenate([[self.id_column, self.timestamp_column], drop, timedeltas, [target]]), axis=1)
        for c in timedeltas:
            if c != target:
                df_X[c] = df[c].dt.seconds

        print("train columns: ")
        print(df_X.columns.values)

        # Choose the target attribute
        if target in timedeltas:
            df_Y = df[target].dt.seconds
        else:
            df_Y = df[target]

        summary = []
        X = df_X.values

        for train_index, test_index, train_interval, test_interval in splitter.split(df_X, df_Y, df[self.id_column], df[self.timestamp_column]):
            X_train, X_test = df_X.loc[train_index], df_X.loc[test_index]
            Y_train, Y_test = df_Y[train_index], df_Y[test_index]
            print("shapes: " + str((X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)))
            
            regressor = RandomForestRegressor(random_state=0,n_jobs=-1)
            regressor.fit(X_train, Y_train)
            test_predict = regressor.predict(X_test)
            
            test_mae = mean_absolute_error(Y_test, test_predict)
            test_mse = mean_squared_error(Y_test, test_predict)

            print(train_interval, test_interval)
            print("test set (MAE / MSE): ", test_mae, test_mse)

            summary = np.append(summary, [test_mae, test_mse, train_interval, test_interval], axis=0)
        
        return summary

    def train_and_evaluate(self, X_train, X_val, X_test, Y_train, Y_val, Y_test, cols_to_remove = [], preprocessing = None):
        # Remove some attributes for training
        train_onehot = X_train.drop(cols_to_remove, axis=1)
        val_onehot = X_val.drop(cols_to_remove, axis=1)
        test_onehot = X_test.drop(cols_to_remove, axis=1)

        # Now, we are going to build estimators for the remaining time
        regressor = RandomForestRegressor(random_state=0,n_jobs=-1)
        regressor.fit(train_onehot, Y_train)

        # Predict
        val_predict = regressor.predict(val_onehot)
        test_predict = regressor.predict(test_onehot)

        # And evaluate the model
        val_mae = mean_absolute_error(Y_val, val_predict)
        val_mse = mean_squared_error(Y_val, val_predict)
        test_mae = mean_absolute_error(Y_test, test_predict)
        test_mse = mean_squared_error(Y_test, test_predict)

        print("validation set (MAE / MSE): ", val_mae, val_mse )
        print("test set (MAE / MSE): ", test_mae, test_mse)

        return val_mae, val_mse, test_mae, test_mse
        
    
    def run_test(self, split_func, drop=[], target = 'remaining_time'):
        """Just a facility to run the default configuration of tests easier"""

        timedeltas = self.df.select_dtypes('timedelta').columns.values
        incidents_X = self.df.drop(np.concatenate([[self.id_column], drop, timedeltas, [target]]), axis=1)
        for c in timedeltas:
            if c != target:
                incidents_X[c] = df[c].dt.seconds

        print("train columns: ")
        print(incidents_X.columns.values)

        # Choose the target attribute
        if target in timedeltas:
            incidents_Y = df[target].dt.seconds
        else:
            incidents_Y = df[target]

        # And the attribute (id) that is used to split the dataset in train, validation and test
        incidents_group = df.groupby(self.id_column)[self.timestamp_column].min().sort_values().index

        X_train, X_test, Y_train, Y_test, group_train, group_test = split_func(incidents_X, incidents_Y, incidents_group, test_size = 0.4)
        X_val, X_test, Y_val, Y_test, group_val, group_test = split_func(X_test, Y_test, group_test, test_size=0.5)

        print("shapes: " + str((X_train.shape, Y_train.shape, X_test.shape, Y_test.shape, X_val.shape, Y_val.shape)))
        train_and_evaluate(X_train, X_val, X_test, Y_train, Y_val, Y_test, cols_to_remove = [])    
    

        