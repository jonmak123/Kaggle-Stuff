# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 20:51:07 2017

@author: Jonathan Mak
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from explore import clean
import pickle
import matplotlib
import matplotlib.pyplot as plt
import time
matplotlib.rcParams['figure.figsize'] = [12, 8]


def load_data(load = False):
    
    print('loading data...')
    
    if load: 
        df = pickle.load(open('processed_data/data.p', 'rb'))
    else:
        df = clean()
    
    print('completed loading data')
        
    return(df)


def drop_cols(df):
    df_ = df.drop(['id', 
                   'pickup_datetime', 
                   'dropoff_datetime',   
                   'avg_speed', 
                   'log_trip_duration', 
                   'trip_duration', 
                   'trip_duration_norm'], axis = 1, errors = 'ignore')
    return(df_)


def crop_train(df_, std = 2):
    crop = df_
#    crop = df_[(abs(df_['pickup_lat_norm']) < std) & \
#              (abs(df_['dropoff_lon_norm']) < std) & \
#              (abs(df_['pickup_lon_norm']) < std) & \
#              (abs(df_['dropoff_lat_norm']) < std) ]
#              (abs(df_['trip_duration_norm']) < std) & \
              
    return(crop)
    

def split(df, val_size):
    
    print('splitting into sets...')
    
#    df = crop_train(df)
    
    ## SPLIT TRAIN AND TEST SETS
    train_x = df[df['log_trip_duration'].isnull() == False][:-val_size]
    train_x = crop_train(train_x)
    train_x = drop_cols(train_x)
    
    train_y = df[df['log_trip_duration'].isnull() == False][:-val_size]
    train_y = crop_train(train_y)
    train_y = train_y['log_trip_duration']
    
    val_x = df[df['log_trip_duration'].isnull() == False][-val_size:]
    val_x = drop_cols(val_x)
    
    val_y = df[df['log_trip_duration'].isnull() == False][-val_size:]['log_trip_duration']
    
    test_x = df[df['log_trip_duration'].isnull() == True].drop('log_trip_duration', axis = 1)
    test_x = drop_cols(test_x)
    
    ## MAKE DMATRICES
    train_batch = xgb.DMatrix(train_x, train_y)
    val_batch = xgb.DMatrix(val_x, val_y)
    test_batch = xgb.DMatrix(test_x)
    
    print('completed splitting')
    
    return(train_batch, val_batch, test_batch)


def train_model(train_batch, val_batch):
    
    print('commence training...')
    
    watchlist = [(train_batch, 'train'), (val_batch, 'val')]
#    watchlist  = [(train_batch,'train')]
    
    eval_result = dict()
    
#    params = {'min_child_weight': 1, 
#              'eta': 0.5, 
#              'colsample_bytree': 0.9, 
#              'max_depth': 6,
#              'subsample': 0.9, 
#              'lambda': 1., 
#              'nthread': -1,
#              'booster' : 'gbtree',
#              'eval_metric': 'rmse',
#              'objective': 'reg:linear',
#              }
    params = {'objective': 'reg:linear',
              'verbose': True, 
              'eval_metric' : 'rmse', 
              'eta' : 0.5, 
              'min_child_weight' : 50
              }

    ## START TRAINING
    model = xgb.train(params = params,
                      dtrain = train_batch, 
                      num_boost_round = 500,
                      early_stopping_rounds = 5,
                      maximize = False,
                      evals = watchlist,
                      evals_result = eval_result)
    
    ## SAVE MODEL
#    model.save_model('model.model')
        
    return(model, eval_result)
       

def error_analysis(df, train_batch, xgb_model, val_size = 100000, show_analysis = False):
    
    ## RECREATE TRAIN_X
    train_x = df[df['log_trip_duration'].isnull() == False][:-val_size]
#    train_x = crop_train(train_x)
        
    if show_analysis:
        ## ADD PREDICTED VALUES
        train_x['pred'] = xgb_model.predict(train_batch)
        
        ## CALCULATE ERROR
        train_x['mse'] = np.power((train_x['pred'] - train_x['log_trip_duration']), 2)

        ## HAVE A LOOK AT DISTRIBUTION OF ERROR, STRANGE BUMP FOR MSE > 12
        plt.hist(train_x['mse'], bins = 200, log = True)
        
        ## ZOOM IN TO SEE THOSE RECORDS
        look = train_x[train_x['mse'] > 12]
        
        ## SEE DISTRIBUTION OF LOG DURATION< SIGNIFICANT BUMP AROUND FROM 10 to 12
        plt.hist(look['log_trip_duration'], bins = 50)
        
        ## ZOOM IN EVEN MORE TO LOOK
        plt.hist(look['log_trip_duration'], bins = 50, range = [10, 12])
    
    ## CREATE FLAG FOR THOSE RECORDS
    train_x['flag'] = train_x['log_trip_duration'].apply(lambda x: 1 if x > 11.25 and x < 11.5 else 0)

    ## DROP IRRELEVANT COLUMNS
    x = drop_cols(train_x).drop('flag', axis = 1)
    y = train_x['flag']
    
    x_train = x.iloc[:-val_size]
    y_train = y[:-val_size]
    train_ = xgb.DMatrix(x_train, y_train)
    
    x_test = x.iloc[-val_size:]
    y_test = y[-val_size:]
    test_ = xgb.DMatrix(x_test, y_test)
    
    return(train_, test_)
    
 
def train_secondary(train_, test_):
    
    watchlist = [(train_, 'train'), (test_, 'val')]
    
    params = {'objective': 'reg:logistic',
              'verbose': True, 
              'eval_metric' : 'auc', 
              'scale_pos_weight' : 780
              }
    
    model_err = xgb.train(params = params,
                      dtrain = train_, 
                      num_boost_round = 50,
#                      early_stopping_rounds = 5,
                      maximize = False,
                      evals = watchlist)
    
    return(model_err)
    
        
if __name__ == '__main__':
    
    ##### LOAD AND MAKE MODEL #####
    df = load_data(load = True)
    train_batch, val_batch, test_batch = split(df, val_size = 100000)
    xgb_model, log = train_model(train_batch, val_batch )

    ##### CREATE SUBMISSION #####  
#    id_ = df[df['log_trip_duration'].isnull() == True]['id']
#    pred = pd.DataFrame({'id': id_, 'trip_duration' : np.exp(xgb_model.predict(test_batch))})
#    pred.to_csv('Submission.csv', index = False)
    
    ##### INVESTIGATE ERROR AND TRY TO CLASSIFY OUTLIERS #####
#    train_, test_ = error_analysis(df, train_batch, xgb_model)
#    model2 = train_secondary(train_, test_)
#    pred_err = model2.predict(test_)
    
    
    
    