#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:53:58 2023

@author: mariohabibfathi
"""

import pandas as pd
import Distance_Metrics as DM


from dtw.dtw import dtw

from tensorflow import keras
import numpy as np
from Constants import UCR_PATH
import DataAug as DA
import Datasets as ds
import FCNEncoder


d = ds.Dataset(UCR_PATH)

path_output = 'outputs/FCNEncoder_multipledata'
result_path = 'outputs/FCNEncoder_multipledata/DTW_resultes/'


datasets = ['SmoothSubspace','SyntheticControl','BeetleFly','BirdChicken',
            'Coffee','UMD','Meat','ItalyPowerDemand','Chinatown','ECG200'
            ,'DistalPhalanxTW','MiddlePhalanxTW','Fungi','Beef','CBF']
# datasets = ['Meat','ItalyPowerDemand','Chinatown','ECG200']
# datasets = [ 'Plane','ArrowHead','Trace','OliveOil','Ham','Herring',
#             'InsectWingbeatSound','Lightning7','ECGFiveDays','Lightning2','Adiac',
#             'Wafer','SwedishLeaf','TwoLeadECG','ShakeGestureWiimoteZ','PickupGestureWiimoteZ',
#             'GesturePebbleZ2','GesturePebbleZ1','Rock']
# datasets = ['BeetleFly']

datasets = ['SmoothSubspace','SyntheticControl','BeetleFly','BirdChicken',
            'Coffee','UMD','Meat','ItalyPowerDemand','Chinatown','ECG200'
            ,'DistalPhalanxTW','MiddlePhalanxTW','Fungi','Beef','CBF',
            'Plane','ArrowHead','Trace','OliveOil','Ham','Herring',
            'InsectWingbeatSound','Lightning7','ECGFiveDays','Lightning2','Adiac',
            'Wafer','SwedishLeaf','TwoLeadECG','ShakeGestureWiimoteZ','PickupGestureWiimoteZ',
            'GesturePebbleZ2','GesturePebbleZ1','Rock']


min_values_index = []
max_values_index = []


for data in datasets:

    xtrain,ytrain,xtest,ytest = d.load_dataset(data)
    
    xtrain = d.znormalisation(xtrain)
    xtest_df1 = d.znormalisation(xtest)
    
    # xtrain = xtrain*4
    pred_df2 = np.load(path_output+'/'+data+'_predicted.npy')
    
    n = xtest_df1.shape[0]
    similarity_list = []
    
    for i in range(n):
        # print(i)
        ts1 = xtest_df1[i]
        ts2 = pred_df2[i]
        dist,_,_,_ = dtw(ts1, ts2, dist= lambda x,y:np.linalg.norm(x-y))
        simi = 1/(1+dist)
        similarity_list.append(simi)
        
    min_sim = np.min(similarity_list)
    print("Min value :",min_sim," index : ",similarity_list.index(min_sim))
    min_values_index.append(similarity_list.index(min_sim))
    if min_sim < 0 :
        similarity_list = [s-min_sim for s in similarity_list]
    max_sim = np.max(similarity_list)
    print("Max value :",max_sim," index : ",similarity_list.index(max_sim))
    max_values_index.append(similarity_list.index(max_sim))
    if max_sim > 1 :
        similarity_list = [s/max_sim for s in similarity_list]
    
    
    
    perce = (np.sum(similarity_list))*100
    print("the similarty percantage between dataset {} is {} percent".format(data,perce))
df = pd.DataFrame(list(zip(datasets,min_values_index,max_values_index)),columns = ['Dataset','min index','max index'])
df.to_csv(result_path+'DTW_resultes.csv')
        
    