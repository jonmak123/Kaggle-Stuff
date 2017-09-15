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


def load_data(load = True):
    
    print('loading data...')
    
    if load: 
        x = pickle.load(open('processed_data/data.p', 'rb'))
    else:
        x = clean()
    
    y = x['log_trip_duration']  
    x = x.drop('log_trip_duration', axis = 1)
    
    rand = np.arange(x.shape[0])
    x = x.iloc[rand]
    y = y.iloc[rand]
    return(x, y)


def create_dummies(batch_i, batch_size, x):
    
    x_ = x.iloc[batch_i : batch_i + batch_size]
    
    list_of_cols = ['vendor_id', 
                    'pickup_cluster', 
                    'dropoff_cluster',
                    'pickup_month',
                    'pickup_weekday',
                    'pickup_hour', 
                    'dropoff_month', 
                    'dropoff_weekday', 
                    'dropoff_hour']
    
    for col in list_of_cols:
        dummies = pd.get_dummies(x[col].astype(str), prefix = col)
        x_ = pd.concat([x_, dummies.iloc[batch_i : batch_i + batch_size]], axis = 1)
        x_.drop(col, axis = 1, inplace = True)
    x_.drop(['id', 'store_and_fwd_flag', 'pickup_datetime', 'dropoff_datetime', 'pickup_day', 'dropoff_day', 'trip_duration'], axis = 1, inplace = True)
    
    return(x_)


def train_model(x, y, epoch = 1, test_prop = 0.01, start_new = True):
    
    print('commence training...')
    
    batch_i = 0
    batch_size = 2000
    test_size = 10000
    
    train_sep = len(x) - test_size
#    test_batch = xgb.DMatrix(create_dummies(train_sep, len(x), x), y[-test_size:])
    
    log = dict()

    
    for ep in range(epoch):
    
        plt.figure()
        plt.title('epoch = ' + str(ep))    
        
        for i in range(round(train_sep/batch_size)):

            print('batch_number:', i)
            
            log[ep] = dict()
            plt.title('epoch = ' + str(ep))
            
            train_batch = xgb.DMatrix(create_dummies(batch_i, batch_size, x), y[batch_i : batch_i + batch_size])
    #        watchlist = [(train_batch, 'train'), (test_batch, 'test')]
            watchlist  = [(train_batch,'train')]
            eval_result = dict()
    #        params = {'min_child_weight': 1, 
    #                  'eta': 0.5, 
    #                  'colsample_bytree': 0.9, 
    #                  'max_depth': 6,
    #                  'subsample': 0.9, 
    #                  'lambda': 1., 
    #                  'nthread': -1,
    #                  'booster' : 'gbtree',
    #                  'eval_metric': 'rmse'
    #                  }
            params = {'objective': 'reg:linear',
                      'verbose': True, 
                      'eval_metric' : 'rmse'
                      }
            
            if i == 0:
                ## START TRAINING
                if start_new:
                    model = xgb.train(params = params,
                                      dtrain = train_batch, 
                                      num_boost_round = 1,
                                      maximize = False,
                                      evals = watchlist,
                                      evals_result = eval_result)
                    params.update({'process_type' : 'update',
                                     'updater' : 'refresh',
                                     'refresh_leaf' : True})
                if start_new == False:
                    params.update({'process_type' : 'update',
                         'updater' : 'refresh',
                         'refresh_leaf' : True})
                    model = xgb.train(params = params,
                              dtrain = train_batch, 
                              num_boost_round = 1,
                              maximize = False,
                              evals = watchlist, 
                              evals_result = eval_result, 
                              xgb_model = 'model.model')
                
            if i > 0:
                ## USE PREVIOUS MODEL TO CONTINUE TRAINING
                model = xgb.train(params = params,
                                  dtrain = train_batch, 
                                  num_boost_round = 1,
                                  maximize = False,
                                  evals = watchlist, 
                                  evals_result = eval_result, 
                                  xgb_model = model)
                
            ## LOG BATCH RESULTS
            log[ep][i] = eval_result
            plt.plot(i, eval_result['train']['rmse'], 'ro')
            plt.show()
            plt.pause(0.05)
            
            ## UPDATE BATCH NUMBER
            batch_i += batch_size
            model.save_model('model.model')
        
        ## EVALUATE PERFORMANCE ON TEST SET AFTER EACH EPOCH
        
    return(model, log)
        
        
if __name__ == '__main__':
    x, y = load_data(load = True)
#    xgb_model, log = train_model(x, y)
    
    
    
    
    
    
    