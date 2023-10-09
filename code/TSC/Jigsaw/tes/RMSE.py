#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 15:37:49 2023

@author: mariohabibfathi
"""

from Constants import UCR_PATH

import Datasets as ds
import Distance_Metrics as DM
import visualization as VS
import classifiers as CL
import LogisticRegression as LR

from sklearn.metrics import accuracy_score
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
import numpy as np
import DataAug as DA
import FCN
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
# import FCNEncoder
# import FCNEncoderDecoder
import os


d = ds.Dataset(UCR_PATH)

datasets = ['SmoothSubspace','UMD','ECG200','Beef']

type_bottle = ['nobottleneck','bottleneck']
# type_bottle = ['bottleneck']
datasets = ['Beef']

segments_indx = []
rmse_score = []
# mean_seg = []
i = 0
for seg_num in range(2,41):
    for permutation_num in range(1,seg_num+1):
        for bot in type_bottle:
            path_output = f'outputs/FCNEncoderDecoder/run4/jigsaw/{seg_num}_Segments/{permutation_num}-permutation/{bot}/'
            if not os.path.exists(path_output):
                # if seg_num == 40 and permutation_num == 30:
                #     os.makedirs(path_output)
                i +=1
                break
            for data in datasets:
                
                xtrain,ytrain,xtest,ytest = d.load_dataset(data)
                
                # xtrain = d.znormalisation(xtrain)
                xtest = d.znormalisation(xtest)
                segments_indx.append(f'{data} with {seg_num} segments for {permutation_num} permutations and {bot}')
                                
                # path_output = f'outputs/FCNEncoderDecoder/run4/jigsaw/{seg_num}_Segments/1-permutation/{bot}/'
                predicted = np.load(path_output+data+'_predicted.npy')
                rmse_score.append(np.sqrt(mean_squared_error(xtest, predicted)))
    
        
# print(i)        
#         mean_seg.append(np.mean(rmse))
#         print(f'mean',np.mean(rmse))



# print(mean_seg.index(min(mean_seg)))            
# print(min(mean_seg))
     
minscoreindex = rmse_score.index(min(rmse_score))
name = segments_indx[minscoreindex]

print(rmse_score.index(min(rmse_score)))            
print(min(rmse_score))
print(name)

print('Worst')

minscoreindex = rmse_score.index(max(rmse_score))
name = segments_indx[minscoreindex]

print(rmse_score.index(max(rmse_score)))            
print(max(rmse_score))
print(name)


# print(len(rmse_score))
        
# df = pd.DataFrame(list(zip(segments_indx,rmse_score)),columns = ['Test Type','RMSE'])
# df.to_csv('outputs/FCNEncoderDecoder/run4/jigsaw/resultes/elsa7/rmse_for_all_segments_all_permutation.csv')












# mean_seg = []
# for seg_num in range(3,41):
#     rmse = []
    
#     for data in datasets:
        
#         xtrain,ytrain,xtest,ytest = d.load_dataset(data)
        
#         xtrain = d.znormalisation(xtrain)
#         xtest = d.znormalisation(xtest)
#         for bot in type_bottle:
            
#             data_type.append(f'{data} with number of {seg_num} Segments and {bot}')
            
            
#             path_output = f'outputs/FCNEncoderDecoder/run4/jigsaw/{seg_num}_Segments/1-permutation/{bot}/'
#             predicted = np.load(path_output+data+'_predicted.npy')
#             rmse.append(np.sqrt(mean_squared_error(xtest, predicted)))

            
#     mean_seg.append(np.mean(rmse))
#     print(f'mean',np.mean(rmse))

# print(mean_seg.index(min(mean_seg)))            
# print(min(mean_seg))