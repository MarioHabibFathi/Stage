#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:35:37 2023

@author: mariohabibfathi
"""

from Constants import UCR_PATH
import Datasets as ds
import visualization as VS
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import matplotlib.pyplot as plt

d = ds.Dataset(UCR_PATH)

# d.Show_Dataset()
# path_output = 'outputs/FCNEncoder_multipledata'
path_image_data = 'images/FCNEncoderDecoder/run4/jigsaw/max_min/'

df = pd.read_csv('outputs/FCNEncoderDecoder/run4/jigsaw/DTW_resultes_for_all_segments_all_permutation.csv',index_col=0)
# ,usecols=['Dataset','min index','max index'])



# datasets = df['Test Type']
min_index = df['min index']
max_index = df['max index']


datasets = ['SmoothSubspace','UMD','ECG200','Beef']
type_bottle = ['nobottleneck','bottleneck']



np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)                 




# sssssssssssssss




i=0
break_points = 0
for seg_num in range(2,41):
    print("We are at segment number ",seg_num)

    for permutation_num in range(1,seg_num+1):
        for bot in type_bottle:
            # seg_num =19
            # permutation_num = 19
            
            path_output = f'outputs/FCNEncoderDecoder/run4/jigsaw/{seg_num}_Segments/{permutation_num}-permutation/{bot}/'

            if not os.path.exists(path_output):
                # if seg_num == 40 and permutation_num == 30:
                #     os.makedirs(path_output)
                break_points +=1
                break

    
            # print("We are at segment number ",seg_num," and permutation number ",permutation_num,f' and {bot}')

            for data in datasets:

                xtrain,ytrain,xtest,ytest = d.load_dataset(data)
                
                # xtrain = d.znormalisation(xtrain)
                xtest = d.znormalisation(xtest)
                predection = np.load(path_output+data+'_predicted.npy')        

                min_y_value = min(min(xtest[min_index[i]]),min(predection[min_index[i]]),
                                  min(xtest[max_index[i]]),min(predection[max_index[i]]))-0.1
                max_y_value = max(max(xtest[min_index[i]]),max(predection[min_index[i]]),
                                  max(xtest[max_index[i]]),max(predection[max_index[i]]))+0.1
                
                plt.plot(xtest[max_index[i]], color="blue", label = "Original")
                plt.plot(predection[max_index[i]], color="green", label = "Predection")
                # plt.plot(predection_3_seg[i], color="red", label = "with bottleneck")
                plt.legend()
                plt.ylim(ymin = min_y_value,ymax = max_y_value)
                plt.title(f'{data} with {seg_num} segments for {permutation_num} permutations \n and {bot} worst DTW for sample {max_index[i]}')
                plt.savefig(path_image_data+'pdf/'+f'{data}_worst_DTW_{seg_num}_segments_{permutation_num}_permutations_with_{bot}.pdf')
                plt.savefig(path_image_data+'png/'+f'{data}_worst_DTW_{seg_num}_segments_{permutation_num}_permutations_with_{bot}.png')

                plt.show()

                plt.plot(xtest[min_index[i]], color="blue", label = "Original")
                plt.plot(predection[min_index[i]], color="green", label = "Predection")
                # plt.plot(predection_3_seg[i], color="red", label = "with bottleneck")
                plt.legend()
                plt.ylim(ymin = min_y_value,ymax = max_y_value)
                plt.title(f'{data} with {seg_num} segments for {permutation_num} permutations \n and {bot} best DTW for sample {min_index[i]}')
                plt.savefig(path_image_data+'pdf/'+f'{data}_best_DTW_{seg_num}_segments_{permutation_num}_permutations_with_{bot}.pdf')
                plt.savefig(path_image_data+'png/'+f'{data}_best_DTW_{seg_num}_segments_{permutation_num}_permutations_with_{bot}.png')

                plt.show()









                print(f'we are at {data} and a segmeent {seg_num} and {permutation_num} {bot} and we have min {min_index[i]} max {max_index[i]} ')

                i += 1

#                 wwwwwwww
                # if i == 10:
                #     ddddddddddddddd



# ssssss







# for data in datasets:
        
#     # for i in range(10):
#         data_name = data.split()[0]
#         xtrain,ytrain,xtest,ytest = d.load_dataset(data_name)
        
#         # xtrain = d.znormalisation(xtrain)
#         xtest = d.znormalisation(xtest)
        
#         # xtrain = xtrain*4
    
        
#         LE = LabelEncoder()
        
#         # ytrain = LE.fit_transform(ytrain)
#         ytest = LE.fit_transform(ytest)
        
#         pred_df2 = np.load(path_output+'/'+data+'_predicted.npy')
        
#         min_y_value = min(min(xtest[min_index[i]]),min(pred_df2[min_index[i]]),
#                           min(xtest[max_index[i]]),min(pred_df2[max_index[i]]))-0.1
#         max_y_value = max(max(xtest[min_index[i]]),max(pred_df2[min_index[i]]),
#                           max(xtest[max_index[i]]),max(pred_df2[max_index[i]]))+0.1
        
        
        
#         vs = VS.Dataset_Visualization()
#         vs.Plot_one(xtest[min_index[i]],title = data + " original min DTW",save=False
#                     ,ylimite=(min_y_value,max_y_value),path = 'images/multipledata/1/')
#         vs.Plot_one(pred_df2[min_index[i]],title = data+" predicted min DTW",save=False
#                     ,ylimite=(min_y_value,max_y_value),path = 'images/multipledata/1/')
#         vs.Plot_one(xtest[max_index[i]],title = data+" original max DTW",save=False
#                     ,ylimite=(min_y_value,max_y_value),path = 'images/multipledata/1/')
#         vs.Plot_one(pred_df2[max_index[i]],title = data+" predicted max DTW",save=False
#                     ,ylimite=(min_y_value,max_y_value),path = 'images/multipledata/1/')
#         # vs.Plot_one(xtest_permutated[0],title = "Permutated for "+data,save=False)
#         i +=1
