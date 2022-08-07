import pandas as pd
import numpy as np
import plotly.express as px
import os
from sklearn.metrics import r2_score
import glob
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('./optiver-realized-volatility-prediction/train.csv')
list_order_book_file_train = glob.glob('./optiver-realized-volatility-prediction/book_train.parquet/*')

#function and definition provided by optiver
def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff() 

#function and definition provided by optiver    
def realized_volatility(series_log_return):
    return np.sqrt(np.sum(series_log_return**2))

#function and definition provided by optiver
def realized_volatility_per_time_id(file_path, prediction_column_name):
    df_book_data = pd.read_parquet(file_path)
    df_book_data['wap'] =(df_book_data['bid_price1'] * df_book_data['ask_size1']+df_book_data['ask_price1'] * df_book_data['bid_size1'])/(df_book_data['bid_size1']+ df_book_data['ask_size1'])
    df_book_data['log_return'] = df_book_data.groupby(['time_id'])['wap'].apply(log_return)
    df_book_data = df_book_data[~df_book_data['log_return'].isnull()]

    #add features for linear regression and lgb model test
    df_book_data['bid_ask_spread'] = df_book_data['ask_price1'] - df_book_data['bid_price1']
    df_book_data['bid_gap'] = df_book_data['bid_price1'] - df_book_data['bid_price2']
    df_book_data['ask_gap'] = df_book_data['ask_price2'] - df_book_data['ask_price1']   
    df_book_data['bidsize_imbalance'] = df_book_data['bid_size1'] / df_book_data['bid_size2']
    df_book_data['asksize_imbalance'] = df_book_data['ask_size1'] / df_book_data['ask_size2']
            
    df_realized_vol_per_stock =  pd.DataFrame(df_book_data.groupby(['time_id'])['log_return'].agg(realized_volatility)).reset_index()
    df_realized_vol_per_stock = df_realized_vol_per_stock.rename(columns = {'log_return':prediction_column_name})
    stock_id = file_path.split('=')[1]
    df_realized_vol_per_stock['row_id'] = df_realized_vol_per_stock['time_id'].apply(lambda x:f'{stock_id}-{x}')

    #features preprocessing for linear regression and lgb model test
    df_book_data_agg = pd.DataFrame(df_book_data.groupby(['time_id'])['log_return', 'wap', 'bid_ask_spread', 'bid_gap', 'ask_gap', 'bidsize_imbalance', 'asksize_imbalance', 'bid_price1', 'ask_price1', 'bid_price2', 'ask_price2', 'bid_size1', 'ask_size1', 'bid_size2', 'ask_size2'].agg(realized_volatility)).reset_index()
    stock_id = file_path.split('=')[1]
    df_book_data_agg['row_id'] = df_book_data_agg['time_id'].apply(lambda x:f'{stock_id}-{x}')   
    return df_realized_vol_per_stock[['row_id',prediction_column_name]], df_book_data_agg[['row_id','log_return', 'wap', 'bid_ask_spread', 'bid_gap', 'ask_gap', 'bidsize_imbalance', 'asksize_imbalance', 'bid_price1', 'ask_price1', 'bid_price2', 'ask_price2', 'bid_size1', 'ask_size1', 'bid_size2', 'ask_size2']]

#function and definition provided by optiver    
def past_realized_volatility_per_stock(list_file, prediction_column_name):
    df_past_realized = pd.DataFrame()
    for file in list_file:
        a, b = realized_volatility_per_time_id(file,prediction_column_name)
        df_past_realized = pd.concat([df_past_realized, a])
    return df_past_realized

df_past_realized_train = past_realized_volatility_per_stock(list_file=list_order_book_file_train, prediction_column_name='pred')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#lgb and linear regression based data preparation
def lgb_dataloader(list_file, prediction_column_name):
    df_lgb = pd.DataFrame()
    for file in list_file:
        a, b = realized_volatility_per_time_id(file,prediction_column_name)
        df_lgb = pd.concat([df_lgb, b])
    return df_lgb

#To calculate the naive prediction(algo provided by optiver)    
train['row_id'] = train['stock_id'].astype(str) + '-' + train['time_id'].astype(str)
train = train[['row_id','target']]
df_joined = train.merge(df_past_realized_train[['row_id','pred']], on = ['row_id'], how = 'left')

R2 = round(r2_score(y_true = df_joined['target'], y_pred = df_joined['pred']),3)
print(f'Performance of the naive prediction: R2 score: {R2}')

import matplotlib.pyplot as plt
plt.scatter(df_joined['target'].values, df_joined['pred'].values)
plt.savefig("1.jpg")

model = LinearRegression()
x = lgb_dataloader(list_order_book_file_train, 'pred').reset_index()
df_joined = train.merge(x, on = ['row_id'], how = 'left')
model.fit(df_joined.drop("row_id",axis='columns').drop("target",axis='columns').fillna(0).values, df_joined['target'].values)
print("linear regression R2 score:", model.score(df_joined.drop("row_id",axis='columns').drop("target",axis='columns').fillna(0).values, df_joined['target'].values))
plt.scatter(model.predict(df_joined.drop("row_id",axis='columns').drop("target",axis='columns').fillna(0).values), df_joined['target'].values)
plt.savefig("2.jpg")

model = lgb.LGBMRegressor()
x = lgb_dataloader(list_order_book_file_train, 'pred').reset_index()
df_joined = train.merge(x, on = ['row_id'], how = 'left')
model.fit(df_joined.drop("row_id",axis='columns').drop("target",axis='columns').fillna(0).values, df_joined['target'].values)
print("lightgbm R2 score:", model.score(df_joined.drop("row_id",axis='columns').drop("target",axis='columns').fillna(0).values, df_joined['target'].values))
plt.scatter(model.predict(df_joined.drop("row_id",axis='columns').drop("target",axis='columns').fillna(0).values), df_joined['target'].values)
plt.savefig("3.jpg")

from lightgbm import plot_importance
plt.figure(figsize=(20,10))
plot_importance(model)
plt.title("Feature Importance")
plt.savefig("feature_importance.jpg")

model = lgb.LGBMRegressor(num_leaves=100)
x = lgb_dataloader(list_order_book_file_train, 'pred').reset_index()
df_joined = train.merge(x, on = ['row_id'], how = 'left')
model.fit(df_joined.drop("row_id",axis='columns').drop("target",axis='columns').fillna(0).values, df_joined['target'].values)
print("lightgbm R2 score (num_leaves=100):", model.score(df_joined.drop("row_id",axis='columns').drop("target",axis='columns').fillna(0).values, df_joined['target'].values))
plt.scatter(model.predict(df_joined.drop("row_id",axis='columns').drop("target",axis='columns').fillna(0).values), df_joined['target'].values)
plt.savefig("4.jpg")
