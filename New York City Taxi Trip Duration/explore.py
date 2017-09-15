# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 14:07:21 2017

@author: jmak
"""

import pandas as pd
import numpy as np
import sklearn
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean
import pickle


##### ORSM CHEATS #####
def cheat(df_):
    
    cheats = pd.concat([pd.read_csv('Cheat/fastest_routes_test.csv'), 
                             pd.read_csv('Cheat/fastest_routes_train_part_1.csv'), 
                             pd.read_csv('Cheat/fastest_routes_train_part_2.csv')])
    df = pd.merge(df, cheats, how = 'left', on = 'id')
    
    return(df)


##### FEATURE ENGINEERING #####
def clean(save = True):
    
    raw_train = pd.read_csv('original/train.csv')
    raw_test = pd.read_csv('original/test.csv')
    
    ## TEMP TRAIN-DEV
    df = pd.concat([raw_train, raw_test], axis = 0)
    
    ## EXTRACT DATETIME FEATURES
    for x in ['pickup_datetime']:
        df[x] = pd.to_datetime(df[x])
        df[x.replace('datetime', 'month')] = df[x].dt.month
        df[x.replace('datetime', 'weekday')] = df[x].dt.weekday
        df[x.replace('datetime', 'day')] = df[x].dt.day
        df[x.replace('datetime', 'hour')] = df[x].dt.hour
        df[x.replace('datetime', 'minute')] = df[x].dt.minute
        
    ## EXTRACT EUCLIDEAN DISTANCE BETWEEN POINTS    
    df['linear_dist'] = ((df['dropoff_latitude'] - df['pickup_latitude']) ** 2 +  (df['dropoff_longitude'] - df['pickup_longitude']) ** 2) ** 0.5   
    
    ## CLEAN STRING
    df['store_and_fwd_flag'].replace({'Y' : 1, 
                                      'N' : 0}, inplace = True)
    
    ## IMPLEMENT HAVERSINE FUNCTION TO CALCULATE ALTERNATIVE DISTANCE
    def arrays_haversine(lats1, lngs1, lats2, lngs2, R=6371):
        lats_delta_rads = np.radians(lats2 - lats1)
        lngs_delta_rads = np.radians(lngs2 - lngs1)
        a = np.sin(lats_delta_rads / 2)**2 + np.cos(lats1) * np.cos(lats2) * np.sin(lngs_delta_rads / 2)**2
        c = 2 * np.arcsin(a**0.5)
        return(R * c)
    
    df['haversine'] = arrays_haversine(df['pickup_latitude'], 
                                       df['pickup_longitude'],
                                       df['dropoff_latitude'], 
                                       df['dropoff_longitude'])
    
    
    ## CALCULATE LINEAR (STRAIGHT LINE) SPEED
    df['avg_speed'] = df['linear_dist'] / df['trip_duration']
    
    ## FIND POLAR BEARING OF TRIP
    def find_polar_bearing(lon2, lon1, lat2, lat1):
        c_theta = np.arctan2(lon2 - lon1, lat2 - lat1) * -180 /  np.pi + 90
        if c_theta >= 360:
            c_theta -= 360
        if c_theta < 0:
            c_theta += 360
        return(c_theta)
    df['direction_polar'] = df.apply(lambda x: find_polar_bearing(x['dropoff_longitude'], 
                                                                  x['pickup_longitude'],
                                                                  x['dropoff_latitude'], 
                                                                  x['pickup_latitude']), axis = 1)
    df['direction_polar'] = (df['direction_polar'] // 10 * 10).astype(int)
    
    df['log_trip_duration'] = np.log(df['trip_duration'] + 1)
    
    def normalise(s):
        return((s - s.mean()) / s.std())
    
    df['pickup_lon_norm'] = normalise(df['pickup_longitude'])
    df['pickup_lat_norm'] = normalise(df['pickup_latitude'])
    df['dropoff_lon_norm'] = normalise(df['dropoff_longitude'])
    df['dropoff_lat_norm'] = normalise(df['dropoff_latitude'])
    df['trip_duration_norm'] = normalise(df['log_trip_duration'])
    
    ## LOAD KMEANS RESULTS
    load_clust = pickle.load(open('./cluster_analysis/cluster.p', 'rb'))
    df['pickup_zone'] = load_clust.predict(df[['pickup_latitude', 'pickup_longitude']])
    df['dropoff_zone'] = load_clust.predict(df[['dropoff_latitude', 'dropoff_longitude']])

    if save:    
        pickle.dump(df, open('processed_data/data.p', 'wb'))
        
    return(df)

    
##### CLUSTER FOR LATER VISUALISATION #####
def clustering(df, n_):
    
    train_pickup = np.insert(np.array(df[['pickup_latitude', 'pickup_longitude']]), 2, 1, axis = 1)
    train_dropoff = np.insert(np.array(df[['dropoff_latitude', 'dropoff_longitude']]), 2, 0, axis = 1)
    train_clust = np.append(train_pickup, train_dropoff, axis = 0)[:, [0, 1]]
    
    ## POOR MAN"S METHOD
    train_clust = train_clust[np.random.choice(train_clust.shape[0], 600000, replace=False)]
    
    print('Running KMeans clustering with n_center =', n_)
    cluster = KMeans(n_clusters = n_).fit(train_clust)
    
    pickle.dump(cluster, open('./cluster_analysis/cluster.p', 'wb'))
    
    return(cluster)


##### USE CLUSTERING RESULTS AND RAW DATA TO PLOT MAP AND DENOTE SEGMENTATION #####
def draw_cluster(clust, df):
    
    ## zip cluster coordinates
    x, y= zip(*clust.cluster_centers_)
    grp = len(x)
    
    ## PLOT SCATTER FOR EVERY ZONE
    plt.figure(figsize = (12,12))
    for i in range(grp):
        lon = df[df['pickup_zone'] == i]['pickup_longitude']
        lat = df[df['pickup_zone'] == i]['pickup_latitude']
        plt.plot(lon, lat, marker = '.', linestyle = '', c = np.random.rand(3,), alpha = 0.9, markersize = 1, label = str(i))
        
    plt.plot(list(y), list(x), c = 'r', marker = '^', linestyle = '', alpha = 1, markersize = 5)
    plt.xlim([-74.05, -73.8])
    plt.ylim([40.6, 40.9])
    plt.title('n_cluster = ' + str(len(list(set(grp))))) 
    

##### MAKE MULTIPLE MAPS WITH CLUSTERING RESULTS AND SAVE #####
def make_multi_maps(df):
    for num in [5, 10, 15, 20, 25, 30, 40, 50, 60, 80]:
        clust = clustering(df, num)
        draw_cluster(clust, df)
        plt.savefig('./cluster_analysis/clustered_map/clust_' + str(num) + '_map.png', dpi = 150)
        plt.close()
    

##### EXPLORATIVE ANALYSIS #####    
def explore(df):

    load_clust = pickle.load(open('./cluster_analysis/cluster.p', 'rb'))
    
    fig, ax = plt.subplots(1, 2)
    ax[0].hist(df['pickup_hour'], bins = 24, rwidth = 0.5)
    ax[0].set_title('Hour')
    ax[1].hist(df['pickup_weekday'], bins = 7, rwidth = 0.5, color = 'g')
    ax[1].set_title('Weekday')
    
    plt.figure()
    sns.boxplot(df, 'trip_duration')
    plt.ylim([0, 2500])
    plt.title('duration hist')
    
    plt.figure()
    plt.plot(df['linear_dist'], df['trip_duration'], 'r.', markersize = 0.5)
    plt.xlim([0, 0.3])    
    plt.ylim([0, 5000])
    plt.title('linear distance v duration')
    
    plt.figure()
    sns.boxplot(data = df, x = 'dropoff_weekday', y = 'trip_duration')
    plt.ylim([0, 2500])
    plt.title('weekday v duration')
    
    plt.figure()
    sns.boxplot(data = df, x = 'passenger_count', y = 'trip_duration')
    plt.ylim([0, 2500])
    plt.title('passenger count v duration')
    df.groupby('passenger_count')['id'].count()
    
    plt.figure()
    sns.boxplot(data = df, x = 'pickup_hour', y = 'trip_duration')
    plt.ylim([0, 2500])
    plt.title('pickup hour v duration')
    
    plt.figure()
    sns.boxplot(data = df, x = 'vendor_id', y = 'trip_duration')
    plt.ylim([0, 2500])
    plt.title('vendor v duration')
    
    grid = sns.FacetGrid(df, col = "pickup_weekday", hue = 'pickup_weekday', col_wrap = 3)
    grid.map(plt.scatter, 'linear_dist', 'trip_duration', s = 0.1)
    grid.set(xlim = (0, 0.3), ylim = (0, 5000))
    plt.title('linear distance v duration v weekday')
    
    plt.figure()
    dist_time = df.groupby(['pickup_cluster', 'dropoff_cluster'], as_index = True)['trip_duration'].mean().reset_index()
    center_dist = pd.DataFrame([[euclidean(x, y) for x in load_clust.cluster_centers_] for y in load_clust.cluster_centers_]).reset_index()
    center_dist = pd.melt(center_dist, id_vars='index', value_vars=center_dist.columns.tolist()[1:]) 
    dist_time = pd.merge(dist_time, center_dist, how = 'left', left_on = ['pickup_cluster', 'dropoff_cluster'], right_on = ['index', 'variable'])
    dist_time['dist_time_ratio'] = dist_time['value'] / dist_time['trip_duration']
    dist_time['ratio_norm'] = (dist_time['dist_time_ratio'] - dist_time['dist_time_ratio'].mean()) / dist_time['dist_time_ratio'].std()
    plt.hist(dist_time['ratio_norm'], range = [0, .3], bins = 30)
    plt.title('average normalised speed')
    
    test = dist_time.sort_values('ratio_norm')
    draw_cluster(load_clust)
    for i in range(0, -20, -1):
        p_, d_ = test.iloc[i, [0, 1]]
        lat_pt, lon_pt = zip(*[ load_clust.cluster_centers_[p_], load_clust.cluster_centers_[d_]])
        plt.plot(lon_pt, lat_pt, 'r-', linewidth = 1)

     
###############################################################################
###############################################################################    
    
if __name__ == '__main__':
    print('Running explore.py...')
#    clustering(df, 120)
    df = clean()
#    make_multi_maps()
#    explore()

    
    
    
    
    