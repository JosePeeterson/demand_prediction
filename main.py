
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import sys

from deepar.dataset.time_series import TimeSeriesTrain, TimeSeriesTest
import pandas as pd
import numpy as np
from deepar.model.learner import DeepARLearner

'''
dem = np.load('1_freq_stoch_nbinom_dem.npy')
df = pd.DataFrame()
df['hour'] = np.arange(1,(len(dem) - 50),1)
df['demand'] = dem[1:(len(dem) - 50)]
df.to_csv('1_f_nbinom_train.csv')

df = pd.DataFrame()
df['hour'] = np.arange((len(dem) - 50),len(dem),1)
df['demand'] = dem[(len(dem) - 50):len(dem)]
df.to_csv('1_f_nbinom_score.csv')

df = pd.DataFrame()
df['hour'] = np.arange((len(dem) - 50),len(dem),1)
df['demand'] = np.empty(50)
df.to_csv('1_f_nbinom_test.csv')
sys.exit()
'''

dem_df = pd.read_csv('C:\Work\WORK_PACKAGE\Demand_forecasting\github\DeepAR\\train_data\\1_f_nbinom_train.csv')
dem_df['hour'] = pd.to_datetime(dem_df['hour'], unit='h') 
print(dem_df['hour'])

dem_ds_one = TimeSeriesTrain(dem_df, target_idx = 3, timestamp_idx = 1, index_col=0,count_data=True)


learner = DeepARLearner(dem_ds_one, verbose=1)


learner.fit(epochs = 5, steps_per_epoch = 30, early_stopping = False,tensorboard=True )


test_df = pd.read_csv('C:\Work\WORK_PACKAGE\Demand_forecasting\github\DeepAR\\test_data\\1_f_nbinom_test.csv')

test_df['hour'] = pd.to_datetime(test_df['hour'], unit='h')

test_ds = TimeSeriesTest(dem_ds_one,test_df,  target_idx = 3)



preds = learner.predict(test_ds, horizon = None, samples = 1, include_all_training = True).reshape(-1)

scores = pd.read_csv('C:\Work\WORK_PACKAGE\Demand_forecasting\github\DeepAR\score\\1_f_nbinom_score.csv')


from sklearn.metrics import mean_squared_error
from math import sqrt

print('\n\n\n',preds[-20:],'\n\n\n')
print('\n\n\n',scores['forecast'][-20:],'\n\n\n')

rms = sqrt(mean_squared_error(scores['demand'][-20:], preds[-20:]))
print(f'rms: {rms}')

print('\n\n\n\n\ndone\n\n\n\n')







