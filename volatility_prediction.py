# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 13:43:41 2024

@author: dbda
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path='D:\BigData_staging\BIGDATAPROYECT'    

#Generate the list of ID
import glob
list_order_book_file_train = glob.glob(path+"/book_train.parquet/*")
list_stock_id = [path.split("=")[1] for path in list_order_book_file_train]
list_stock_id = np.sort(np.array([int(i) for i in list_stock_id]))

#%%
#Defining important function
def wap(df,bid_price,ask_price,bid_size,ask_size):
    return (df[bid_price]*df[ask_size]+df[ask_price]*df[bid_size])/(df[bid_size]+df[ask_size])

def log_return(wap):
    return np.log(wap).diff()

def volatility(log_return):
    return np.sqrt(np.sum(log_return**2))

def preprocess_book(file_path):
    stock_id = file_path.split("=")[1]
    book_df=pd.read_parquet(file_path)
    
    book_df['micro_price1'] = wap(book_df,'bid_price1','ask_price1','bid_size1','ask_size1')
    book_df['log_return1'] = log_return(book_df['micro_price1'])
    
    book_df['micro_price2'] = wap(book_df,'bid_price2','ask_price2','bid_size2','ask_size2')
    book_df['log_return2'] = log_return(book_df['micro_price2'])
    
    # Spread normalized by "mean price"
    # book_df['spread'] = 2*(book_df['ask_price1'] - book_df['bid_price1'])/(book_df['ask_price1'] + book_df['bid_price1'])
    # # Low market depth could indicate a sharp future price movement in case of aggressive buy or sell 
    # book_df['ask_depth'] = book_df['ask_size1'] + book_df['ask_size2'] 
    # book_df['bid_depth'] = book_df['bid_size1'] + book_df['bid_size2']
    
    # book_df['volume_imbalance'] = np.abs(book_df['ask_size1'] - book_df['bid_size1'])*2/(book_df['ask_size1']+book_df['bid_size1'])
    
   
    aggregate = {
        'log_return1' : volatility,
        'log_return2' : volatility,
        # 'spread' : np.mean,
        # 'ask_depth' : np.mean,
        # 'bid_depth' : np.mean,
        # 'volume_imbalance' : np.mean
    }

    preprocessed_df = book_df.groupby('time_id').agg(aggregate)
    preprocessed_df = preprocessed_df.rename(columns={'log_return1':'volatility1',
                                                      'log_return2':'volatility2'})
    # preprocessed_df_last_300 = book_df[book_df['seconds_in_bucket']>300].groupby('time_id').agg(aggregate)
    # preprocessed_df_last_300 = preprocessed_df_last_300.rename(columns={'log_return1':'realized_volatility1_last_300',
    #                                                            'log_return2':'realized_volatility2_last_300',
    #                                                            'spread':'spread_last_300',
    #                                                            'ask_depth':'ask_depth_last_300',
    #                                                            'bid_depth':'bid_depth_last_300',
    #                                                            'volume_imbalance':'volume_imbalance_last_300'})
    
    preprocessed_df.reset_index(inplace=True)
    # preprocessed_df_last_300.reset_index(inplace=True)
    preprocessed_df['row_id'] = preprocessed_df['time_id'].apply(lambda x:f'{stock_id}-{x}')
    preprocessed_df.drop('time_id', axis=1, inplace=True)
    # preprocessed_df_last_300['row_id'] = preprocessed_df_last_300['time_id'].apply(lambda x:f'{stock_id}-{x}')
    # preprocessed_df_last_300.drop('time_id', axis=1, inplace=True)
    # return preprocessed_df.merge(preprocessed_df_last_300, how='left', on='row_id')
    return preprocessed_df

def relative_fluctuation(data):
    return np.std(data)/np.mean(data)

def preprocess_trade(file_path):
    stock_id = file_path.split("=")[1]
    trade_df = pd.read_parquet(file_path)
    aggregate = {
        'price': relative_fluctuation,
        'size': relative_fluctuation,
        'order_count': relative_fluctuation
    }
    preprocessed_df = trade_df.groupby('time_id').agg(aggregate)
    preprocessed_df = preprocessed_df.rename(columns={'price':'trade_price_fluc', 
                                                      'size':'size_fluc', 
                                                      'order_count':'orders_fluc'})
    preprocessed_df.reset_index(inplace=True)
    preprocessed_df['row_id'] = preprocessed_df['time_id'].apply(lambda x:f'{stock_id}-{x}')
    preprocessed_df.drop('time_id', axis=1, inplace=True)
    
    # preprocessed_df_last_300 = trade_df[trade_df['seconds_in_bucket']>300].groupby('time_id').agg(aggregate)
    # preprocessed_df_last_300 = preprocessed_df_last_300.rename(columns={'price':'trace_price_rv_last_300', 
    #                                                                     'size':'volume_last_300', 
    #                                                                     'order_count':'number_of_orders_last_300'})
    # preprocessed_df_last_300.reset_index(inplace=True)
    # preprocessed_df_last_300['row_id'] = preprocessed_df_last_300['time_id'].apply(lambda x:f'{stock_id}-{x}')
    # preprocessed_df_last_300.drop('time_id', axis=1, inplace=True)
    # return preprocessed_df.merge(preprocessed_df_last_300, how='left', on='row_id')
    return preprocessed_df


# prueba=preprocess_trade(path+"/trade_train.parquet/stock_id=0")

# pruebabook=preprocess_book(path+"/book_train.parquet/stock_id=0")

from joblib import Parallel, delayed

def prep_merge_trade_book(list_stock_id,state='train'):
    trade_book_df = pd.DataFrame()
    def job(stock_id,state=state):
        if state=='train':
            book_path = path+"/book_train.parquet/stock_id="+str(stock_id)
            trade_path = path+"/trade_train.parquet/stock_id="+str(stock_id)
        elif state=='test':
            book_path = path+"/book_test.parquet/stock_id="+str(stock_id)
            trade_path = path+"/trade_test.parquet/stock_id="+str(stock_id)  
        else:
            return print('Insert correct state: train/test')
        book_df = preprocess_book(book_path)
        trade_df = preprocess_trade(trade_path)
        temp_df = book_df.merge(trade_df, how='left', on='row_id')
        return(pd.concat([trade_book_df, temp_df]))
    
    trade_book_df = Parallel(n_jobs=-1, verbose=1) (delayed(job)(stock_id) for stock_id in list_stock_id)
    trade_book_df = pd.concat(trade_book_df)
    
    train_df = pd.read_csv(path+'/train.csv')
    train_df['row_id'] = train_df['stock_id'].astype(str) + '-' + train_df['time_id'].astype(str)
    train_df.drop(['stock_id', 'time_id'], axis=1, inplace=True)

    return trade_book_df.merge(train_df, how='left', on='row_id').reset_index(drop=True)

trade_book_df = prep_merge_trade_book(list_stock_id)
trade_book_df.to_csv('trade_book_df.csv')


#ALERT
#We have 19 nan values we will deleted
trade_book_df.dropna(inplace=True)

#%%
#We see the correlation of the differents features
import seaborn as sns
corr = trade_book_df[['volatility1', 'volatility2', 'trade_price_fluc', 'size_fluc',
       'orders_fluc', 'target']].corr()
fig = plt.figure(figsize=(5, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm")


#%%
#Now we analize the data with knn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold

def rmspe(y_true, y_pred):
    loss = np.sqrt(np.mean(np.square((y_true-y_pred)/y_true)))
    return loss

# df = pd.read_csv("./trade_book_df.csv", index_col=0)
y = trade_book_df['target']
X = trade_book_df.drop(['target', 'row_id'], axis=1)

knn=KNeighborsRegressor()
params={'n_neighbors':np.arange(1,10)}
kfold = KFold(n_splits=5, shuffle= True, random_state= 23)
rgcv=RandomizedSearchCV(knn, param_distributions=params,n_iter=2,cv=kfold, random_state=1,scoring='r2')

rgcv.fit(X,y)
print(rgcv.best_params_)
print(rgcv.best_score_)

#%%
#Now we create the test data
list_test_id = [0]

trade_test_df = prep_merge_trade_book(list_test_id,'test')
trade_test_df.to_csv('trade_test_df.csv')


X_test=trade_test_df.drop(['target', 'row_id'],axis=1)
y_pred=rgcv.predict(X_test)




#%%
ss = pd.read_csv(path+"\sample_submission.csv")
ss['target']=y_pred
ss.to_csv('optiver_knn.csv',index=False)