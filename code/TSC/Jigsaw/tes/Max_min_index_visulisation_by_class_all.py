#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 16:14:08 2023

@author: mariohabibfathi
"""


from Constants import UCR_PATH
import Datasets as ds
import visualization as VS
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json
import matplotlib.pyplot as plt
import os

d = ds.Dataset(UCR_PATH)

# d.Show_Dataset()
path_image_data = 'images/FCNEncoderDecoder/run4/jigsaw/max_min_by_class/'
df = pd.read_csv('outputs/FCNEncoderDecoder/run4/jigsaw/DTW_resultes_by_class_for_all_segments_all_permutation.csv',index_col=0)
# ,usecols=['Dataset','min index','max index'])



# datasets = df.iloc[:,0]
min_indexes = df['min index by class']
max_indexes = df['max index by class']

# aaa = min_indexes[i]
# aaa = aaa.strip('][').split(', ')
# # min_indexes[i] = min_indexes[i].strip('][').split(', ')
# print(aaa[0])


datasets = ['SmoothSubspace','UMD','ECG200','Beef']
type_bottle = ['nobottleneck','bottleneck']



np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   

# print(min_indexes[0][0])

# min_indexes[0] = json.loads(min_indexes[0])

# print(min_indexes[0][0])

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
                xtest = d.znormalisation(xtest)
                predection = np.load(path_output+data+'_predicted.npy') 


                LE = LabelEncoder()
                
                ytrain = LE.fit_transform(ytrain)
                ytest = LE.fit_transform(ytest)
                num_classes = len(np.unique(ytest))
                min_indexes[i] = json.loads(min_indexes[i])
                max_indexes[i] = json.loads(max_indexes[i])
                # print(num_classes)
                for class_number in range(num_classes):
                    # print(class_number)
                    # print(f'min index ONLY {min_indexes[i]}')
                    # print(f'min index {min_indexes[i][class_number]}')
                    # print(f'max index ONLY {max_indexes[i]}')
                    # print(f'max index {max_indexes[i][class_number]}')
                    
                    
                    # print(xtest[min_indexes[i][class_number]])
                    min_y_value = min(min(xtest[min_indexes[i][class_number]]),min(predection[min_indexes[i][class_number]]),
                                      min(xtest[max_indexes[i][class_number]]),min(predection[max_indexes[i][class_number]]))-0.1
                    max_y_value = max(max(xtest[min_indexes[i][class_number]]),max(predection[min_indexes[i][class_number]]),
                                      max(xtest[max_indexes[i][class_number]]),max(predection[max_indexes[i][class_number]]))+0.1                
                    plt.plot(xtest[max_indexes[i][class_number]], color="blue", label = "Original")
                    plt.plot(predection[max_indexes[i][class_number]], color="green", label = "Predection")
                    # plt.plot(predection_3_seg[i], color="red", label = "with bottleneck")
                    plt.legend()
                    plt.ylim(ymin = min_y_value,ymax = max_y_value)
                    plt.title(f'{data} with {seg_num} segments for {permutation_num} permutations \n and {bot} worst DTW for sample {max_indexes[i][class_number]} and class {class_number}')
                    plt.savefig(path_image_data+'pdf/'+f'{data}_worst_DTW_class_{class_number}_{seg_num}_segments_{permutation_num}_permutations_with_{bot}.pdf')
                    plt.savefig(path_image_data+'png/'+f'{data}_worst_DTW_class_{class_number}_{seg_num}_segments_{permutation_num}_permutations_with_{bot}.png')
    
                    plt.show()
    
                    plt.plot(xtest[min_indexes[i][class_number]], color="blue", label = "Original")
                    plt.plot(predection[min_indexes[i][class_number]], color="green", label = "Predection")
                    # plt.plot(predection_3_seg[i], color="red", label = "with bottleneck")
                    plt.legend()
                    plt.ylim(ymin = min_y_value,ymax = max_y_value)
                    plt.title(f'{data} with {seg_num} segments for {permutation_num} permutations \n and {bot} best DTW for sample {min_indexes[i][class_number]} and class {class_number}')
                    plt.savefig(path_image_data+'pdf/'+f'{data}_best_DTW_class_{class_number}_{seg_num}_segments_{permutation_num}_permutations_with_{bot}.pdf')
                    plt.savefig(path_image_data+'png/'+f'{data}_best_DTW_class_{class_number}_{seg_num}_segments_{permutation_num}_permutations_with_{bot}.png')
    
                    plt.show()
                    
                i+=1
#                 ffffffffffff


# ffffffffffffffff





# for data in datasets:
        
#     # for i in range(10):
        
#         xtrain,ytrain,xtest,ytest = d.load_dataset(data)
        
#         # xtrain = d.znormalisation(xtrain)
#         xtest = d.znormalisation(xtest)
        
#         # xtrain = xtrain*4
    
        
#         LE = LabelEncoder()
        
#         # ytrain = LE.fit_transform(ytrain)
#         ytest = LE.fit_transform(ytest)
        
#         pred_df2 = np.load(path_output+'/'+data+'_predicted.npy')
#         num_classes = len(np.unique(ytest))

#         min_indexes[i] = json.loads(min_indexes[i])
#         max_indexes[i] = json.loads(max_indexes[i])
#         for class_number in range(num_classes):
#             # print(min_indexes[i][class_number])
#             # print(type(min_indexes[i]))
#             # min_indexes[i] = min_indexes[i].strip('][').split(', ')
            
#             # print(min_indexes[i][class_number])
#             # print(type(min_indexes[i]))
#             # saas = min_indexes[i][class_number]
            
#             # print(xtest[saas])
#             min_y_value = min(min(xtest[min_indexes[i][class_number]]),min(pred_df2[min_indexes[i][class_number]]),
#                               min(xtest[max_indexes[i][class_number]]),min(pred_df2[max_indexes[i][class_number]]))-0.1
#             max_y_value = max(max(xtest[min_indexes[i][class_number]]),max(pred_df2[min_indexes[i][class_number]]),
#                               max(xtest[max_indexes[i][class_number]]),max(pred_df2[max_indexes[i][class_number]]))+0.1
        
        
#             # plt.suptitle(data +" " +str(class_number))

#             # plt.subplot(2, 2, 1)
#             # plt.plot(xtest[min_indexes[i][class_number]], color="blue")
#             # # plt.title(data +" " +str(class_number)+ " min dtw original" )
#             # plt.title("min dtw original" )

#             # plt.ylim(min_y_value,max_y_value)

#             # plt.subplot(2, 2, 2)
#             # plt.plot(pred_df2[min_indexes[i][class_number]], color="blue")
#             # # plt.title(data +" " +str(class_number)+ " min dtw predicted" )
#             # plt.title("min dtw predicted" )

#             # plt.ylim(min_y_value,max_y_value)

#             # plt.subplot(2, 2, 3)
#             # plt.plot(xtest[max_indexes[i][class_number]], color="blue")
#             # # plt.title(data +" " +str(class_number)+ " max dtw original" )
#             # plt.title("max dtw original" )

#             # plt.ylim(min_y_value,max_y_value)
            
#             # plt.subplot(2, 2, 4)
#             # plt.plot(pred_df2[max_indexes[i][class_number]], color="blue")
#             # # plt.title(data +" " +str(class_number)+ " max dtw predicted" )
#             # plt.title("max dtw predicted" )

#             # plt.ylim(min_y_value,max_y_value)
#             # plt.subplots_adjust(left=0.1,
#             #         bottom=0.1,
#             #         right=0.9,
#             #         top=0.9,
#             #         wspace=0.4,
#             #         hspace=0.4)
#             # plt.savefig('images/DTW4/'+data +"_" +str(class_number)+'.pdf')
#             # # plt.subplot_tool()
#             # plt.show()

#             vs = VS.Dataset_Visualization()
#             vs.Plot_one(xtest[min_indexes[i][class_number]],title =data +"_" +str(class_number)+ "_min_dtw_original",save=True
#                         ,ylimite=(min_y_value,max_y_value),path = 'images/DTW1/')
#             vs.Plot_one(pred_df2[min_indexes[i][class_number]],title =data +"_" +str(class_number)+ "_min_dtw_predicted",save=True
#                         ,ylimite=(min_y_value,max_y_value),path = 'images/DTW1/')
#             vs.Plot_one(xtest[max_indexes[i][class_number]],title =data +"_" +str(class_number)+ "_max_dtw_original",save=True
#                         ,ylimite=(min_y_value,max_y_value),path = 'images/DTW1/')
#             vs.Plot_one(pred_df2[max_indexes[i][class_number]],title =data +"_" +str(class_number)+ "_max_dtw_predicted",save=True
#                         ,ylimite=(min_y_value,max_y_value),path = 'images/DTW1/')
#         # vs.Plot_one(xtest_permutated[0],title = "Permutated for "+data,save=True)
#         i +=1