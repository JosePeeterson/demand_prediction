

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import sys

from deepar.dataset.time_series import TimeSeriesTrain, TimeSeriesTest
import pandas as pd
import numpy as np
from deepar.model.learner import DeepARLearner


sunspots_df = pd.read_csv('C:\Work\WORK_PACKAGE\Demand_forecasting\github\DeepAR\\train_data\learningData.csv')
sunspots_df['year'] = pd.to_datetime(np.cumsum(sunspots_df['year']), unit='s')
#print(sunspots_df['year'])
sunspots_ds_one = TimeSeriesTrain(sunspots_df, target_idx = 4, timestamp_idx = 1, index_col=0)


learner = DeepARLearner(sunspots_ds_one, verbose=1)

learner.fit(epochs = 10, steps_per_epoch = 10, early_stopping = False,tensorboard=True )


test_df = pd.read_csv('C:\Work\WORK_PACKAGE\Demand_forecasting\github\DeepAR\\test_data\learningData.csv')

test_df['year'] = pd.to_datetime(np.cumsum(test_df['year']), unit='s')

test_ds = TimeSeriesTest(sunspots_ds_one,test_df,  target_idx = 4)

preds = learner.predict(test_ds, horizon = None, samples = 1, include_all_training = True).reshape(-1)

scores = pd.read_csv('C:\Work\WORK_PACKAGE\Demand_forecasting\github\DeepAR\score\learningData.csv')


from sklearn.metrics import mean_squared_error
from math import sqrt

print('\n\n\n',preds[-20:],'\n\n\n')
print('\n\n\n',scores['sunspots'][-20:],'\n\n\n')

rms = sqrt(mean_squared_error(scores['sunspots'][-20:], preds[-20:]))
print(f'rms: {rms}')

print('\n\n\n\n\ndone\n\n\n\n')









