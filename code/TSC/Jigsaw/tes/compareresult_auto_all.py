#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 10:09:52 2023

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
# import FCNEncoder
import os

d = ds.Dataset(UCR_PATH)

# path_output = 'outputs/FCNEncoder_multipledata'
# result_path = 'outputs/FCNEncoder_multipledata/DTW_resultes/'


# datasets = ['SmoothSubspace','SyntheticControl','BeetleFly','BirdChicken',
#             'Coffee','UMD','Meat','ItalyPowerDemand','Chinatown','ECG200'
#             ,'DistalPhalanxTW','MiddlePhalanxTW','Fungi','Beef','CBF']
# datasets = ['Meat','ItalyPowerDemand','Chinatown','ECG200']
# datasets = [ 'Plane','ArrowHead','Trace','OliveOil','Ham','Herring',
#             'InsectWingbeatSound','Lightning7','ECGFiveDays','Lightning2','Adiac',
#             'Wafer','SwedishLeaf','TwoLeadECG','ShakeGestureWiimoteZ','PickupGestureWiimoteZ',
#             'GesturePebbleZ2','GesturePebbleZ1','Rock']
# datasets = ['BeetleFly']

# datasets = ['SmoothSubspace','SyntheticControl','BeetleFly','BirdChicken',
#             'Coffee','UMD','Meat','ItalyPowerDemand','Chinatown','ECG200'
#             ,'DistalPhalanxTW','MiddlePhalanxTW','Fungi','Beef','CBF',
#             'Plane','ArrowHead','Trace','OliveOil','Ham','Herring',
#             'InsectWingbeatSound','Lightning7','ECGFiveDays','Lightning2','Adiac',
#             'Wafer','SwedishLeaf','TwoLeadECG','ShakeGestureWiimoteZ','PickupGestureWiimoteZ',
#             'GesturePebbleZ2','GesturePebbleZ1','Rock']

datasets = ['SmoothSubspace','UMD','ECG200','Beef']
type_bottle = ['nobottleneck','bottleneck']



# path_output_first_part = 'outputs/FCNEncoderDecoder/run3/jigsaw/'
# continue_path_output = '_Segments/1-permutation/nobottleneck/'

# path_result = '_Segments/1-permutation/DTW_result/'

min_values_index = []
max_values_index = []
min_values = []
max_values = []
avg_values = []
range_value = []
segments_indx = []
i = 0
for seg_num in range(2,41):
    print("We are at segment number ",seg_num)

    for permutation_num in range(1,seg_num+1):
        for bot in type_bottle:
            path_output = f'outputs/FCNEncoderDecoder/run4/jigsaw/{seg_num}_Segments/{permutation_num}-permutation/{bot}/'

            if not os.path.exists(path_output):
                # if seg_num == 40 and permutation_num == 30:
                #     os.makedirs(path_output)
                i +=1
                break

    
            print("We are at segment number ",seg_num," and permutation number ",permutation_num,f' and {bot}')
    # path_output = path_output_first_part + str(seg_num)+continue_path_output
    # result_path = path_output_first_part + str(seg_num)+path_result
    # if not os.path.exists(result_path):
    #     os.makedirs(result_path)
    # print(result_path)
    
            
            for data in datasets:
                print(data)
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
                    # simi = 1/(1+dist)
                    similarity_list.append(dist)
                    
                min_sim = np.min(similarity_list)
                print("Min value :",min_sim," index : ",similarity_list.index(min_sim))
                min_values_index.append(similarity_list.index(min_sim))
                min_values.append(min_sim)
                # if min_sim < 0 :
                #     similarity_list = [s-min_sim for s in similarity_list]
                max_sim = np.max(similarity_list)
                print("Max value :",max_sim," index : ",similarity_list.index(max_sim))
                max_values_index.append(similarity_list.index(max_sim))
                max_values.append(max_sim)
                avg_values.append(np.mean(similarity_list))
                range_value.append(max_sim - min_sim)
                segments_indx.append(f'{data} with {seg_num} segments for {permutation_num} permutations and {bot}')
                # if max_sim > 1 :
                #     similarity_list = [s/max_sim for s in similarity_list]
        
        
        
        # perce = (np.sum(similarity_list))*100
        # print("the similarty percantage between dataset {} is {} percent".format(data,perce))
    df = pd.DataFrame(list(zip(segments_indx,min_values,max_values,avg_values,range_value,min_values_index,max_values_index)),
                      columns = ['Test Type','DTW min values','DTW max values','DTW avg values','DTW range values','min index','max index'])
    df.to_csv('outputs/FCNEncoderDecoder/run4/jigsaw/resultes/elsa7/DTW_resultes_for_all_segments_all_permutation.csv')
        
    