import numpy as np
import pandas as pd
from experiments import compute_weights

class TimeCaseSplit:
    def __init__(self, train_size, train_freq, test_freq=pd.DateOffset(0), test_periods=1, train_start = None, threshold=0, sliding=True):
        if test_freq == 0 and test_periods > 1:
            raise ValueError('Invalid combination of test_freq and test_periods')
            
        self.train_start = train_start
        self.train_size = train_size
        self.train_freq = train_freq
        self.test_freq = test_freq
        self.test_periods = test_periods
        self.threshold = threshold
        self.sliding = sliding
        
    def split(self, X, y, group, timestamp):
        grouped = pd.concat([group, timestamp], keys=['group', 'timestamp'], axis=1).groupby('group')['timestamp'].transform('max')
        if self.train_start is None:
            self.train_start = grouped.min()
        max_timestamp = timestamp.max()
            
        dates = pd.date_range(self.train_start, grouped.max(), freq=self.train_freq)
        for i in range(0, len(dates)):
            if self.sliding:
                train_interval = pd.Interval(left=dates[i], right=dates[i] + self.train_size, closed='both')            
            else:
                train_interval = pd.Interval(left=dates[0], right = dates[i] + self.train_size, closed='both')
            train = group[(grouped>=train_interval.left) & (grouped<=train_interval.right)]
            
            if len(train.unique()) < self.threshold:
                continue
            
            for j in range(0, self.test_periods):
                test_interval = pd.Interval(left=(train_interval.right + j * self.test_freq),
                                            right=((train_interval.right + (j+1)*self.test_freq) if (not (self.test_freq == pd.DateOffset(0))) else max_timestamp), 
                                            closed='both')
                #test = group[(grouped>=test_interval.left) & (grouped<=test_interval.right)]
                test = group[(timestamp>=test_interval.left) & (timestamp<=test_interval.right)]
                if len(test.unique()) < self.threshold:
                    continue
                yield train.index.values, test.index.values, train_interval, test_interval
                

class NumberCaseSplit:
    def __init__(self, train_size, train_freq, test_freq=0, test_periods=1, sliding=True, threshold=0, sampling=False):
        if test_freq == 0 and test_periods > 1:
            raise ValueError('Invalid combination of test_freq and test_periods')
            
        self.train_size = train_size
        self.train_freq = train_freq
        self.test_freq = test_freq
        self.test_periods = test_periods
        self.sliding = sliding
        self.sampling = sampling
        self.threshold = threshold
        
    def split(self, X, y, group, timestamp):
        grpby = pd.concat([group, timestamp], keys=['group', 'timestamp'], axis=1).groupby('group')['timestamp']
        max_by_id = grpby.max().sort_values()
        grouped = grpby.transform('max').sort_values()
        
        max_timestamp = timestamp.max()
        time_sorted = timestamp.sort_values()

        d_size = len(max_by_id)
        current = 0

        intervals = pd.interval_range(start=0, end=d_size, freq = self.train_freq)
        for i in range(0, len(intervals)):
            right = intervals[i].left + self.train_size
            if right >= d_size:
                right = d_size - 1
                
            if self.sliding:
                train_interval = pd.Interval(left=max_by_id.iloc[intervals[i].left],
                                             right=max_by_id.iloc[right],
                                             closed='both')
                mbi_subset = max_by_id.iloc[intervals[i].left:right]
            else:
                train_interval = pd.Interval(left=max_by_id.iloc[intervals[0].left],
                                             right=max_by_id.iloc[right],
                                             closed='both')
                mbi_subset = max_by_id.iloc[intervals[0].left:right]

            if self.sampling:
                print('Sampling from: '+str(len(mbi_subset))+ ' in '+str(i+1))
                cw = compute_weights(i+1)
                weights = np.repeat([cw[0]], self.train_size)
                if  i > 0:
                    middle_weights = np.repeat(cw[1:i], self.train_freq)
                    end_weights = np.repeat([cw[i]], len(mbi_subset) - (self.train_size + (i-1)*self.train_freq))
                    weights = np.concatenate((weights, middle_weights, end_weights))

                mbi_subset = mbi_subset.sample(self.train_size, random_state=0, weights=weights)

            train = pd.merge(group, mbi_subset, right_index=True, left_on=group.name)[group.name]
#            train = group[(grouped>=train_interval.left) & (grouped<=train_interval.right)]
            if len(train.unique()) < self.threshold:
                continue
            
            train_interval_right = intervals[i].left + self.train_size
            for j in range(0, self.test_periods):
                if self.test_freq > 0:
                    right_test = train_interval_right + ((j+1)*self.test_freq)
                else:
                    right_test = d_size - 1
                
                if right_test >= d_size:
                    right_test = d_size - 1
                    
                if train_interval_right + j * self.test_freq >= d_size:
                    break;
                
                test_interval = pd.Interval(left=(max_by_id.iloc[train_interval_right + j * self.test_freq]),
                                            right=(max_by_id.iloc[right_test]), 
                                            closed='both')
                #test = group[(grouped>=test_interval.left) & (grouped<=test_interval.right)]
                test = group[(timestamp>=test_interval.left) & (timestamp<=test_interval.right)]
                if len(test.unique()) < self.threshold:
                    continue

                yield train.index.values, test.index.values, train_interval, test_interval

