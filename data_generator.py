from copyreg import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from scipy.stats import nbinom
import pandas as pd





dem_len = 560


def dem_hr(hr, n, p,v,len):

    dem_hr = np.array([])
    for i in range(2000):
        d =  v + np.random.negative_binomial(n, p, 1)
        z = np.array([0]*(hr-1))
        dem_hr =  np.append(dem_hr, d)
        dem_hr =  np.append(dem_hr, z)

    dem_hr = dem_hr[:len]

    return dem_hr

 
def gen_data(len):

    n1 = 9
    p1 = 0.5

    d8 = dem_hr(4, n1, p1, 0,len)
    dem = np.array(d8,dtype=np.float32)

    shape = 1/n1
    mean = (1-p1)/(p1*shape)
    var = mean+(mean**2)*shape
    print(mean,shape,var) # 0.75 0.3333333333333333

    plt.plot(dem)
    plt.show()

    return dem


dem = gen_data(len=dem_len)

np.save('1_freq_stoch_nbinom_dem',dem)




dem = np.load('1_freq_stoch_nbinom_dem.npy')
df = pd.DataFrame()
df['hour'] = np.arange(1,(len(dem) - 58),1)
df['demand'] = dem[1:(len(dem) - 58)]
df['forecast'] = dem[2:(len(dem) - 57)]
df.to_csv('1_f_nbinom_train.csv')

df = pd.DataFrame()
df['hour'] = np.arange((len(dem) - 58),len(dem),1)
df['demand'] = dem[(len(dem) - 58):len(dem)]
tempdf = dem[(len(dem) - 57):len(dem)]
df['forecast'] = np.append(tempdf,0)
df.to_csv('1_f_nbinom_score.csv')

df = pd.DataFrame()
df['hour'] = np.arange((len(dem) - 58),len(dem),1)
df['demand'] = dem[(len(dem) - 58):len(dem)]
df.to_csv('1_f_nbinom_test.csv')



























































# dem_len = 560


# def dem_hr(hr, n, p,v,len):

#     dem_hr = np.array([])
#     for i in range(2000):
#         d =  v + np.random.negative_binomial(n, p, 1)
#         z = np.array([0]*(hr-1))
#         dem_hr =  np.append(dem_hr, d)
#         dem_hr =  np.append(dem_hr, z)

#     dem_hr = dem_hr[:len]

#     return dem_hr


# def gen_data(len):

#     dzero = np.zeros(len)
#     n1 = 4
#     n2 = 20

#     p1 = 0.9
#     p2 = 0.75

#     d8 = dem_hr(4, n1, p1, 0,len)
#     d4 = dem_hr(8, n2, p2,0,len)

#     dall =  dzero + d8
#     dsub = dall - d4
#     dem = np.where(dsub>=0,d8,d4)


#     dem = np.array(dem,dtype=np.float32)
#     return dem


# dem = gen_data(len=dem_len)

# np.save('2_freq_stoch_nbinom_dem',dem)


# n1 = 4
# n2 = 20

# p1 = 0.9
# p2 = 0.75

# shape = 1/n2
# mean = (1-p2)/(p2*shape)
# print(mean,shape) # 0.75 0.3333333333333333

# plt.plot(dem)
# plt.show()































