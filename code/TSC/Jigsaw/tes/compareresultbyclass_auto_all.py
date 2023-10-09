#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 11:29:08 2023

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
from sklearn.preprocessing import LabelEncoder
import os


d = ds.Dataset(UCR_PATH)

# path_output = 'outputs/FCNEncoder'
# result_path = 'outputs/FCNEncoder_multipledata/DTW_resultes/'

# datasets = ['SmoothSubspace','SyntheticControl','BeetleFly','BirdChicken',
#             'Coffee','UMD','Meat','ItalyPowerDemand','Chinatown','ECG200'
#             ,'DistalPhalanxTW','MiddlePhalanxTW','Fungi','Beef','CBF',
#             'Plane','ArrowHead','Trace','OliveOil','Ham','Herring',
#             'InsectWingbeatSound','Lightning7','ECGFiveDays','Lightning2','Adiac',
#             'Wafer','SwedishLeaf','TwoLeadECG','ShakeGestureWiimoteZ','PickupGestureWiimoteZ',
#             'GesturePebbleZ2','GesturePebbleZ1','Rock']

# datasets = ['SmoothSubspace']

datasets = ['SmoothSubspace','UMD','ECG200','Beef']
type_bottle = ['nobottleneck','bottleneck']



# path_output_first_part = 'outputs/FCNEncoderDecoder/run3/jigsaw/'
# continue_path_output = '_Segments/1-permutation/bottleneck/'

# path_result = '_Segments/1-permutation/DTW_result/'

min_values_indexes = []
max_values_indexes = []
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

            for data in datasets:
            
                print(data)
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
                dataset_class_min_value = []
                dataset_class_max_value = []
                dataset_class_avg_value = []
                dataset_class_range_value = []
                for i in range(num_classes):
                    test_class = xtest_df1[ytest==i]
                    pred_class = pred_df2[ytest==i]
                    n = test_class.shape[0]
                    similarity_list = []
                    for j in range(n):
                        ts1 = test_class[j]
                        ts2 = pred_class[j]
                        dist,_,_,_ = dtw(ts1, ts2, dist= lambda x,y:np.linalg.norm(x-y))
                        # simi = 1/(1+dist)
                        similarity_list.append(dist)
                    min_sim_index = similarity_list.index(np.min(similarity_list))
                    max_sim_index = similarity_list.index(np.max(similarity_list))
                    min_sim_value = np.min(similarity_list)
                    max_sim_value = np.max(similarity_list)
                    
                    
                    min_class_value = test_class[min_sim_index]
                    max_class_value = test_class[max_sim_index]
                    
                    print(min_sim_index)
                    # print(min_class_value)
                    # print('min : ',np.where(xtest_df1==min_class_value))
                    # print('min : ',np.where(xtest_df1==min_class_value)[0])
                    # print('min index: ',np.where(xtest_df1==min_class_value)[0][0])
                    # print('max index: ',np.where(xtest_df1==max_class_value)[0][0])
                    print(f'min index {min_sim_index} and value of {min_sim_value}')
                    print(f'max index {max_sim_index} and value of {max_sim_value}')
            
            
            
                    dataset_class_min_indexes.append(np.where(xtest_df1==min_class_value)[0][0])
                    dataset_class_max_indexes.append(np.where(xtest_df1==max_class_value)[0][0])
                    # dataset_class_min_indexes.append(min_sim_index)
                    # dataset_class_max_indexes.append(max_sim_index)
                    dataset_class_min_value.append(min_sim_value)
                    dataset_class_max_value.append(max_sim_value)
                    dataset_class_avg_value.append(np.mean(similarity_list))
                    dataset_class_range_value.append(max_sim_value - min_sim_value)
                    
                min_values_indexes.append(dataset_class_min_indexes)
                max_values_indexes.append(dataset_class_max_indexes)
                min_values.append(dataset_class_min_value)
                max_values.append(dataset_class_max_value)
                avg_values.append(dataset_class_avg_value)
                range_value.append(dataset_class_range_value)
                segments_indx.append(f'{data} with {seg_num} segments for {permutation_num} permutations and {bot}')

                
    # df = pd.DataFrame(list(zip(datasets,min_values_indexes,max_values_indexes)),columns = ['Dataset','min index','max index'])
    df = pd.DataFrame(list(zip(segments_indx,min_values,max_values,avg_values,range_value,min_values_indexes,max_values_indexes)),
                      columns = ['Test Type','DTW min values by class','DTW max values by class','DTW avg values by class','DTW range values by class','min index by class','max index by class'])
    df.to_csv('outputs/FCNEncoderDecoder/run4/jigsaw/resultes/elsa7/DTW_resultes_by_class_for_all_segments_all_permutation.csv')