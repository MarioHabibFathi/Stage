#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 09:46:14 2023

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
from sklearn.preprocessing import LabelEncoder



d = ds.Dataset(UCR_PATH)

path_output = 'outputs/FCNEncoder'
result_path = 'outputs/FCNEncoder_multipledata/DTW_resultes/'

datasets = ['SmoothSubspace','SyntheticControl','BeetleFly','BirdChicken',
            'Coffee','UMD','Meat','ItalyPowerDemand','Chinatown','ECG200'
            ,'DistalPhalanxTW','MiddlePhalanxTW','Fungi','Beef','CBF',
            'Plane','ArrowHead','Trace','OliveOil','Ham','Herring',
            'InsectWingbeatSound','Lightning7','ECGFiveDays','Lightning2','Adiac',
            'Wafer','SwedishLeaf','TwoLeadECG','ShakeGestureWiimoteZ','PickupGestureWiimoteZ',
            'GesturePebbleZ2','GesturePebbleZ1','Rock']

# datasets = ['SmoothSubspace']
min_values_indexes = []
max_values_indexes = []

for data in datasets:

    xtrain,ytrain,xtest,ytest = d.load_dataset(data)
    
    xtrain = d.znormalisation(xtrain)
    xtest_df1 = d.znormalisation(xtest)
    
    # xtrain = xtrain*4
    pred_df2 = np.load(path_output+'/'+data+'_predicted.npy')

    LE = LabelEncoder()
    
    ytrain = LE.fit_transform(ytrain)
    ytest = LE.fit_transform(ytest)
    num_classes = len(np.unique(ytest))


    dataset_class_min_indexes = []
    dataset_class_max_indexes = []
    for i in range(num_classes):
        test_class = xtest_df1[ytest==i]
        pred_class = pred_df2[ytest==i]
        n = test_class.shape[0]
        similarity_list = []
        for j in range(n):
            ts1 = test_class[j]
            ts2 = pred_class[j]
            dist,_,_,_ = dtw(ts1, ts2, dist= lambda x,y:np.linalg.norm(x-y))
            simi = 1/(1+dist)
            similarity_list.append(simi)
        min_sim_index = similarity_list.index(np.min(similarity_list))
        max_sim_index = similarity_list.index(np.max(similarity_list))
        
        min_class_value = test_class[min_sim_index]
        max_class_value = test_class[max_sim_index]
        
        print('min : ',np.where(xtest_df1==min_class_value)[0][0])
        print('max : ',np.where(xtest_df1==max_class_value)[0][0])

        dataset_class_min_indexes.append(np.where(xtest_df1==min_class_value)[0][0])
        dataset_class_max_indexes.append(np.where(xtest_df1==max_class_value)[0][0])

        
    min_values_indexes.append(dataset_class_min_indexes)
    max_values_indexes.append(dataset_class_max_indexes)
    print(data)
df = pd.DataFrame(list(zip(datasets,min_values_indexes,max_values_indexes)),columns = ['Dataset','min index','max index'])
df.to_csv(result_path+'DTW_resultes_by_class.csv')





