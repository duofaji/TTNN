# -*- coding: utf-8 -*-
"""
"""

import numpy as np
from tqdm import tqdm
import pandas as pd
from geopy.distance import geodesic

#%%
def r2(y_label,y_predict):
    
    """ 
    
    To calculate the R-squared parameters.
    
    Parameters
    ----------
    y_label: npy
        The npy file of the depth or distance label.
    
    y_predict: npy
        The npy file of the depth or distance prediction.
           
    Returns
    --------        
    r_squared: float
        The calculated r_squared.
    
    """
    
    SS_R = np.sum((y_label-np.mean(y_label))*(y_predict-np.mean(y_predict)))
    SS_T1 = np.sum((y_label-np.mean(y_label))**2)
    SS_T2 = np.sum((y_predict-np.mean(y_predict))**2)
    r_squared = SS_R/(SS_T1*SS_T2)**0.5
    return r_squared

def evaluate(dep_predict_dir='sampleData/dep_predict.npy',
             dis1_predict_dir='sampleData/dis1_predict.npy',
             dis2_predict_dir='sampleData/dis2_predict.npy',
             dis3_predict_dir='sampleData/dis3_predict.npy',
             dis4_predict_dir='sampleData/dis4_predict.npy',
             metadata1='sampleData/metadata1.csv',
             metadata2='sampleData/metadata2.csv',
             metadata3='sampleData/metadata3.csv',
             metadata4='sampleData/metadata4.csv'):
    
    """ 
    
    To evaluate model performance on test data.
    
    Parameters
    ----------
    dep_predict_dir: str
        Path to the npy file of the predicted focal depths.
    
    dis1_predict_dir: str
        Path to the npy file of the predicted epicenter distances corresponding to the first station.
        
    dis2_predict_dir: str
        Path to the npy file of the predicted epicenter distances corresponding to the second station.
        
    dis3_predict_dir: str
        Path to the npy file of the predicted epicenter distances corresponding to the third station.
            
    dis4_predict_dir: str
        Path to the npy file of the predicted epicenter distances corresponding to the fourth station.
            
    metadata1: str
        Path to the npy file of the metadata corresponding to the observation from the first station.
        
    metadata2: str
        Path to the npy file of the metadata corresponding to the observation from the second station.
        
    metadata3: str
        Path to the npy file of the metadata corresponding to the observation from the third station.
        
    metadata4: str
        Path to the npy file of the metadata corresponding to the observation from the fourth station.
           
    Returns
    --------        
    dep_mean_error: float
        The mean error of the focal depth.
    
    dep_mae: float
        The mean absolute error of the focal depth.
    
    dep_std: float
        The standard deviation of the error of the focal depth.
    
    dep_r2: float
        The R-squared of the focal depth.
    
    dis_mean_error: float
        The mean error of the epicenter distance.
    
    dis_mae: float
        The mean absolute error of the epicenter distance.
    
    dis_std: float
        The standard deviation of the error of the epicenter distance.
    
    dis_r2: float
        The R-squared of the epicenter distance.
    
    """  
    
    ##import metadata
    waveform1_metadata = pd.read_csv(metadata1)
    waveform2_metadata = pd.read_csv(metadata2)
    waveform3_metadata = pd.read_csv(metadata3)
    waveform4_metadata = pd.read_csv(metadata4)
    
    ##label information
    dep_label = np.array(waveform1_metadata['source_depth_km'])
    dis1_label = np.array(waveform1_metadata['source_distance_km'])
    dis2_label = np.array(waveform2_metadata['source_distance_km'])
    dis3_label = np.array(waveform3_metadata['source_distance_km'])
    dis4_label = np.array(waveform4_metadata['source_distance_km'])
    
    ##import prediction
    dep_predict = np.load(dep_predict_dir).squeeze()
    dis1_predict = np.load(dis1_predict_dir).squeeze()
    dis2_predict = np.load(dis2_predict_dir).squeeze()
    dis3_predict = np.load(dis3_predict_dir).squeeze()
    dis4_predict = np.load(dis4_predict_dir).squeeze()
    
    ##evaluation of depth
    dep_mean_error = np.mean(dep_label - dep_predict)
    dep_mae = np.mean(np.abs(dep_label - dep_predict))
    dep_std = np.std(dep_label - dep_predict)
    dep_r2 = r2(dep_label, dep_predict)
    
    ##evaluation of epicenter distance
    dis_predict = np.concatenate((dis1_predict, dis2_predict, dis3_predict, dis4_predict))
    dis_label = np.concatenate((dis1_label, dis2_label, dis3_label, dis4_label))
    
    dis_mean_error = np.mean(dis_label - dis_predict)
    dis_mae = np.mean(np.abs(dis_label - dis_predict))
    dis_std = np.std(dis_label - dis_predict)
    dis_r2 = r2(dis_label, dis_predict)
    
    return dep_mean_error, dep_mae, dep_std, dep_r2, dis_mean_error, dis_mae, dis_std, dis_r2

def fun(lat1,lon1,lat2,lon2,lat3,lon3,lat4,lon4,dis1,dis2,dis3,dis4,eq_lat_init,eq_lon_init):
    
    """ 
    
    To calculate the latitude and longitude of the earthquake epicenter by grid search method.
    
    Parameters
    ----------
    lat1,lon1,lat2,lon2,lat3,lon3,lat4,lon4: npy
        The npy variable of the latitudes and longitudes of the four stations.
    
    dis1,dis2,dis3,dis4: npy
        The npy variable of the predicted epicenter distance corresponding to the four stations.
    
    eq_lat_init,eq_lon_init: npy
        The initial npy variable of the earthquake latitude and longitude.
           
    Returns
    --------        
    lat_pre, lon_pre: npy
        The predicted latitude and longitude of the earthquake epicenter.
    
    """
    
    dis1_pre = np.zeros((200,200))
    dis2_pre = np.zeros((200,200))
    dis3_pre = np.zeros((200,200))
    dis4_pre = np.zeros((200,200))
    lat_u = np.arange(eq_lat_init-1,eq_lat_init+1,0.01)
    lon_u = np.arange(eq_lon_init-1,eq_lon_init+1,0.01)
    for i in range(len(lat_u)):
        for j in range(len(lon_u)):
            dis1_pre[i,j] = geodesic((lat_u[i], lon_u[j]), (lat1, lon1)).kilometers
            dis2_pre[i,j] = geodesic((lat_u[i], lon_u[j]), (lat2, lon2)).kilometers
            dis3_pre[i,j] = geodesic((lat_u[i], lon_u[j]), (lat3, lon3)).kilometers
            dis4_pre[i,j] = geodesic((lat_u[i], lon_u[j]), (lat4, lon4)).kilometers
    
    error = np.abs(dis1_pre - dis1) + np.abs(dis2_pre - dis2) + np.abs(dis3_pre - dis3) + np.abs(dis4_pre - dis4)
    
    min_index  = np.where(error == np.min(error))
    
    lat_pre = lat_u[min_index[0]]
    lon_pre = lon_u[min_index[1]]
    
    return lat_pre, lon_pre

def calculate_epicenter(dep_predict_dir='sampleData/dep_predict.npy',
                        dis1_predict_dir='sampleData/dis1_predict.npy',
                        dis2_predict_dir='sampleData/dis2_predict.npy',
                        dis3_predict_dir='sampleData/dis3_predict.npy',
                        dis4_predict_dir='sampleData/dis4_predict.npy',
                        metadata1='sampleData/metadata1.csv',
                        metadata2='sampleData/metadata2.csv',
                        metadata3='sampleData/metadata3.csv',
                        metadata4='sampleData/metadata4.csv'):
    
    """ 
    
    To calculate the latitudes and longtitudes of the earthquake epicenters.
    
    Parameters
    ----------
    dep_predict_dir: str
        Path to the npy file of the predicted focal depths.
    
    dis1_predict_dir: str
        Path to the npy file of the predicted epicenter distances corresponding to the first station.
        
    dis2_predict_dir: str
        Path to the npy file of the predicted epicenter distances corresponding to the second station.
        
    dis3_predict_dir: str
        Path to the npy file of the predicted epicenter distances corresponding to the third station.
            
    dis4_predict_dir: str
        Path to the npy file of the predicted epicenter distances corresponding to the fourth station.
            
    metadata1: str
        Path to the npy file of the metadata corresponding to the observation from the first station.
        
    metadata2: str
        Path to the npy file of the metadata corresponding to the observation from the second station.
        
    metadata3: str
        Path to the npy file of the metadata corresponding to the observation from the third station.
        
    metadata4: str
        Path to the npy file of the metadata corresponding to the observation from the fourth station.
           
    Returns
    --------        
    eq_lat_pre, eq_lon_pre: npy
        The predicted latitude and longitude of the earthquake epicenter.
    
    """  
    
    ##Read the station's latitude and longitude.
    station1_lat = np.array(pd.read_csv(metadata1)['receiver_latitude'])
    station1_lon = np.array(pd.read_csv(metadata1)['receiver_longitude'])
    station2_lat = np.array(pd.read_csv(metadata2)['receiver_latitude'])
    station2_lon = np.array(pd.read_csv(metadata2)['receiver_longitude'])
    station3_lat = np.array(pd.read_csv(metadata3)['receiver_latitude'])
    station3_lon = np.array(pd.read_csv(metadata3)['receiver_longitude'])
    station4_lat = np.array(pd.read_csv(metadata4)['receiver_latitude'])
    station4_lon = np.array(pd.read_csv(metadata4)['receiver_longitude'])
    
    dep_predict = np.load(dep_predict_dir)
    dis1_predict = np.load(dis1_predict_dir)
    dis2_predict = np.load(dis2_predict_dir)
    dis3_predict = np.load(dis3_predict_dir)
    dis4_predict = np.load(dis4_predict_dir)
    
    eq_lat_pre = np.zeros((dep_predict.shape[0],))
    eq_lon_pre = np.zeros((dep_predict.shape[0],))
    
    eq_lat_init = (station1_lat+station2_lat+station3_lat+station4_lat)/4
    eq_lon_init = (station1_lon+station2_lon+station3_lon+station4_lon)/4
    
    for i in tqdm(range(dep_predict.shape[0])):
        
        lat, lon = fun(station1_lat[i],station1_lon[i],station2_lat[i],station2_lon[i],station3_lat[i],station3_lon[i],
                       station4_lat[i],station4_lon[i],dis1_predict[i],dis2_predict[i],dis3_predict[i],dis4_predict[i], eq_lat_init[i], eq_lon_init[i])
        
        if len(lat) > 1:
            lat = np.mean(lat)
            lon = np.mean(lon)
            
        eq_lat_pre[i] = lat
        eq_lon_pre[i] = lon
        
    return eq_lat_pre, eq_lon_pre