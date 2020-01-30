import numpy as np
import pandas as pd
from experiments import compute_weights

class TimeCaseSplit:
    def __init__(self, train_size, train_freq, test_freq=pd.DateOffset(0), test_periods=1, train_start = None, threshold=0):
        if test_freq == 0 and test_periods > 1:
            raise ValueError('Invalid combination of test_freq and test_periods')
            
        self.train_start = train_start
        self.train_size = train_size
        self.train_freq = train_freq
        self.test_freq = test_freq
        self.test_periods = test_periods
        self.threshold = threshold
        
    def split(self, X, y, group, timestamp, strategy):
        grpby = pd.concat([group, timestamp], keys=['group', 'timestamp'], axis=1).groupby('group')['timestamp']
        max_by_id = grpby.max().sort_values()
        grouped = grpby.transform('max')

        if self.train_start is None:
            self.train_start = grouped.min()
        max_timestamp = timestamp.max()
            
        dates = pd.date_range(self.train_start, grouped.max(), freq=self.train_freq)
        for i in range(0, len(dates)):
            train_interval = pd.Interval(left=dates[0], 
                                         right = dates[i] + self.train_size, 
                                         closed='both')
            
            train_subset = max_by_id[(max_by_id>=train_interval.left) & (max_by_id<=train_interval.right)]
            train_subset, train_interval = strategy.filter(train_subset, train_interval)

            train = pd.merge(group, train_subset, right_index=True, left_on=group.name)[group.name]
            
            if len(train.unique()) < self.threshold:
                continue
            
            for j in range(0, self.test_periods):
                test_interval = pd.Interval(left=(train_interval.right + j * self.test_freq),
                                            right=((train_interval.right + (j+1)*self.test_freq) if (not (self.test_freq == pd.DateOffset(0))) else max_timestamp), 
                                            closed='right')
                #test = group[(grouped>=test_interval.left) & (grouped<=test_interval.right)]
                test = group[(timestamp>test_interval.left) & (timestamp<=test_interval.right)]
                if len(test.unique()) < self.threshold:
                    continue
                yield train.index.values, test.index.values, train_interval, test_interval


def to_time_interval(interval, max_timestamp):
    return pd.Interval(left = max_timestamp.iloc[interval.left],
                        right = max_timestamp.iloc[interval.right],
                        closed=interval.closed)

class CummulativeStrategy:
    def filter(self, max_timestamp, interval):
        return max_timestamp, interval

class NonCummulativeStrategy:
    def __init__(self, train_size):
        super().__init__()
        self.train_size = train_size

    def filter(self, max_timestamp, interval):        
        if isinstance(self.train_size, pd.DateOffset):
            left = max(interval.left, interval.right - self.train_size)
            train_interval = pd.Interval(left = left,
                                         right = interval.right,
                                         closed = 'both')
            result = max_timestamp[(max_timestamp >= train_interval.left) & (max_timestamp <= train_interval.right)]
        else: 
            right = len(max_timestamp) - 1
            left = max(0, right - self.train_size)
            train_interval = pd.Interval(left = max_timestamp.iloc[left],
                                            right = interval.right,
                                            closed = 'both')
            result = max_timestamp.iloc[left:right]

        return result, train_interval


class SamplingStrategy:
    '''Strategy based on sampling the training set with different weights so that
    it is more likely to include the samples of a certain period.

    Parameters
    ----------
    train_size : int
        Maximum size of instances in the training set obtained with this strategy
    weights : (n_periods, )
        Weights applied for each period. Periods will be homogeneously distributed
        throughout the training set
    '''
    def __init__(self, train_size, weights):
        super().__init__()

        self.train_size = train_size
        self.weights = weights

    def filter(self, max_timestamp, interval):
        max_size = min(self.train_size, len(max_timestamp))
        num_intervals = len(self.weights)
        interval_size = len(max_timestamp)//num_intervals
        size_intervals = [interval_size for i in range(0,num_intervals - 1)]
        size_intervals.append(len(max_timestamp)-(interval_size*(num_intervals-1)))
        cw = np.repeat(self.weights, size_intervals)

        return max_timestamp.sample(max_size, random_state=0, weights=cw), interval

class DriftStrategy:
    '''Strategy based on filtering the training set based on drifts detected in
    it. It will filter out samples after a minimum number of samples (determined
    by threshold) occur after the last drift.

    Parameters
    ----------
    drifts : (n_drifts,)
        List with the moments in time where a drift is detected

    threshold : int
        Minimum number of instances after the drift is detected
    '''
    def __init__(self, drifts, threshold):
        super().__init__()

        self.drifts = drifts
        self.threshold = threshold

    def filter(self, max_timestamp, interval):
        subset = max_timestamp
        train_interval = interval
        if len(max_timestamp) >= self.threshold:
            min_left = len(max_timestamp) - self.threshold
            min_time = max_timestamp.iloc[min_left]
            for drift in sorted(self.drifts, reverse=True):
                if min_time > drift:
                    subset = max_timestamp[max_timestamp > drift]
                    train_interval = pd.Interval(left=drift.tz_convert(interval.right.tz),
                                                right=interval.right,
                                                closed='both')
                    break

        
        return subset, train_interval


class CombineStrategy:
    def __init__(self, strategies):
        super().__init__()

        self.strategies = strategies

    def filter(self, max_timestamp):
        result = max_timestamp
        for s in self.strategies:
            result, intervals = s.filter(result)

        return result, intervals

class NumberCaseSplit:
    """Predictive monitoring cross-validator based on the number of cases

    Provides train/test indices to split process event log data samples
    in train/test sets. Like in time series, in each split, test indices 
    must be higher than before, and thus shuffling in cross validator is 
    inappropriate.

    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them. The main
    difference with TimeSeriesSplit is that in this case the split must
    also take the case id into account, so it is kind of a mix between a
    TimeSeriesSplit and a GroupKFold.

    Parameters
    ----------
    train_size : int
        Size of the training set (in number of instances)
    train_step : int
        Size (in number of instances) of each step of the training set in 
        each iteration
    test_freq : int, default = 0
        Size of the testing set (in number of instances) in each iteration. 
        If test_freq=0, then the testing set includes the remaining elements.
    test_periods : int, default = 1
        Number of testing sets included in each iteration 
    threshold : int, default = 0
        Minimum number of instances to include in each training or test set        
    """
    def __init__(self, train_size, train_step, test_freq=0, test_periods=1, threshold=0):
        if test_freq == 0 and test_periods > 1:
            raise ValueError('Invalid combination of test_freq and test_periods')
            
        self.train_size = train_size
        self.train_step = train_step
        self.test_freq = test_freq
        self.test_periods = test_periods
        self.threshold = threshold
        
    def split(self, X, y, group, timestamp, strategy):
        """Generate indices to split data into training and test set
        by applying a strategy for building a new predictive model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data array.
        y : ndarray of shape (n_samples,)        
            The target labels.
        group : ndarray of shape (n_samples,)
            The case id.
        timestamp : ndarray of shape (n_samples,)
            The timestamp of each event.    
        strategy : strategy
            The strategy to be used for selecting the training set
        """
        grpby = pd.concat([group, timestamp], keys=['group', 'timestamp'], axis=1).groupby('group')['timestamp']
        max_by_id = grpby.max().sort_values()
        # grouped = grpby.transform('max').sort_values()
        
        #max_timestamp = timestamp.max()
        #time_sorted = timestamp.sort_values()

        d_size = len(max_by_id)

        intervals = pd.interval_range(start=0, end=d_size, freq = self.train_step)
        for i in range(0, len(intervals)):
            train_interval_left = intervals[0].left
            train_interval_right = intervals[i].left + self.train_size
            if train_interval_right >= d_size:
                train_interval_right = d_size - 1

            train_subset = max_by_id.iloc[train_interval_left:train_interval_right]
            train_interval = pd.Interval(left = max_by_id.iloc[train_interval_left],
                                            right = max_by_id.iloc[train_interval_right],
                                            closed = 'both')

            mbi_subset, train_interval = strategy.filter(train_subset, train_interval)

            train = pd.merge(group, mbi_subset, right_index=True, left_on=group.name)[group.name]
#            train = group[(grouped>=train_interval.left) & (grouped<=train_interval.right)]
            if len(train.unique()) < self.threshold:
                continue
            
            for j in range(0, self.test_periods):
                test_interval_left = train_interval_right + j * self.test_freq

                if self.test_freq > 0:
                    test_interval_right = train_interval_right + ((j+1)*self.test_freq)
                    if test_interval_right >= d_size:
                        test_interval_right = d_size - 1
                else:
                    test_interval_right = d_size - 1
                
                    
                if train_interval_right + j * self.test_freq >= d_size:
                    break
                
                test_interval = pd.Interval(left=(max_by_id.iloc[test_interval_left]),
                                            right=(max_by_id.iloc[test_interval_right]), 
                                            closed='both')
                #test_subset = max_by_id.iloc[test_interval_left, test_interval_right]

                #test = group[(grouped>=test_interval.left) & (grouped<=test_interval.right)]
                test = group[(timestamp>=test_interval.left) & (timestamp<=test_interval.right)]
                #test = pd.merge(group, test_subset, right_index=True, left_on=group.name)[group.name]

                if len(test.unique()) < self.threshold:
                    continue

                yield train.index.values, test.index.values, train_interval, test_interval

