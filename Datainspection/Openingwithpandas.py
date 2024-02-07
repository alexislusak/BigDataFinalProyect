# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 19:05:26 2023

@author: Group 10
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%%
#Charging all
book_train = pd.read_parquet('D:\BigData_staging\BIGDATAPROYECT/book_train.parquet')
book_train.info()

#%%
#Charging only 0
book_train_0 = pd.read_parquet('D:\BigData_staging\BIGDATAPROYECT/book_train.parquet/stock_id=0/c439ef22282f412ba39e9137a3fdabac.parquet')
book_train_0.info()

#%% 
#Charging somes
import glob
subset_paths = glob.glob('D:\BigData_staging\BIGDATAPROYECT/book_train.parquet/stock_id=11*/*')
book_train_subset = pd.read_parquet(subset_paths)
book_train_subset.info()

#%%
#Defining WAP function
def wap(df,bid_price,ask_price,bid_size,ask_size):
    return (df[bid_price]*df[ask_size]+df[ask_price]*df[bid_size])/(df[bid_size]+df[ask_size])

def log_returns(wap):
    return np.log(wap).diff()

def volatility(log_return):
    return np.sqrt(np.sum(log_return**2))


df=book_train_0[book_train_0['time_id']==5]
#Calculating WAP
df['wap1']=wap(df,'bid_price1','ask_price1','bid_size1','ask_size1')


#Ploting the WAP function
plt.figure('wap')
plt.title('WAP funtion order 0 - second in bucket 5')
plt.plot(df['seconds_in_bucket'],df['wap1'])
plt.xlabel('Seconds in bucket')
plt.ylabel('WAP')

#Calculating log return
df['wapLog']=df['wap1'].agg(log_returns)

#Ploting the log return
plt.figure('log')
plt.title('log returns funtion order 0 - second in bucket 5')
plt.plot(df['seconds_in_bucket'],df['wapLog'])
plt.xlabel('Seconds in bucket')
plt.ylabel('log(returns)')

#Calculating volatility
vol = volatility(df['wapLog'])
print('Volatility for stock_id 0 on time_id 5:',vol)


#%%




