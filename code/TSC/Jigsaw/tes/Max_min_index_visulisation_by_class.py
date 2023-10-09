#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 11:38:43 2023

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

d = ds.Dataset(UCR_PATH)

# d.Show_Dataset()
path_output = 'outputs/FCNEncoder'

df = pd.read_csv('outputs/FCNEncoder/DTW_resultes_by_class.csv',index_col=0)
# ,usecols=['Dataset','min index','max index'])



datasets = df.iloc[:,0]
min_indexes = df.iloc[:,1]
max_indexes = df.iloc[:,2]

i = 0

# aaa = min_indexes[i]
# aaa = aaa.strip('][').split(', ')
# # min_indexes[i] = min_indexes[i].strip('][').split(', ')
# print(aaa[0])




for data in datasets:
        
    # for i in range(10):
        
        xtrain,ytrain,xtest,ytest = d.load_dataset(data)
        
        xtrain = d.znormalisation(xtrain)
        xtest = d.znormalisation(xtest)
        
        # xtrain = xtrain*4
    
        
        LE = LabelEncoder()
        
        ytrain = LE.fit_transform(ytrain)
        ytest = LE.fit_transform(ytest)
        
        pred_df2 = np.load(path_output+'/'+data+'_predicted.npy')
        num_classes = len(np.unique(ytest))

        min_indexes[i] = json.loads(min_indexes[i])
        max_indexes[i] = json.loads(max_indexes[i])
        for class_number in range(num_classes):
            # print(min_indexes[i][class_number])
            # print(type(min_indexes[i]))
            # min_indexes[i] = min_indexes[i].strip('][').split(', ')
            
            # print(min_indexes[i][class_number])
            # print(type(min_indexes[i]))
            # saas = min_indexes[i][class_number]
            
            # print(xtest[saas])
            min_y_value = min(min(xtest[min_indexes[i][class_number]]),min(pred_df2[min_indexes[i][class_number]]),
                              min(xtest[max_indexes[i][class_number]]),min(pred_df2[max_indexes[i][class_number]]))-0.1
            max_y_value = max(max(xtest[min_indexes[i][class_number]]),max(pred_df2[min_indexes[i][class_number]]),
                              max(xtest[max_indexes[i][class_number]]),max(pred_df2[max_indexes[i][class_number]]))+0.1
        
        
            # plt.suptitle(data +" " +str(class_number))

            # plt.subplot(2, 2, 1)
            # plt.plot(xtest[min_indexes[i][class_number]], color="blue")
            # # plt.title(data +" " +str(class_number)+ " min dtw original" )
            # plt.title("min dtw original" )

            # plt.ylim(min_y_value,max_y_value)

            # plt.subplot(2, 2, 2)
            # plt.plot(pred_df2[min_indexes[i][class_number]], color="blue")
            # # plt.title(data +" " +str(class_number)+ " min dtw predicted" )
            # plt.title("min dtw predicted" )

            # plt.ylim(min_y_value,max_y_value)

            # plt.subplot(2, 2, 3)
            # plt.plot(xtest[max_indexes[i][class_number]], color="blue")
            # # plt.title(data +" " +str(class_number)+ " max dtw original" )
            # plt.title("max dtw original" )

            # plt.ylim(min_y_value,max_y_value)
            
            # plt.subplot(2, 2, 4)
            # plt.plot(pred_df2[max_indexes[i][class_number]], color="blue")
            # # plt.title(data +" " +str(class_number)+ " max dtw predicted" )
            # plt.title("max dtw predicted" )

            # plt.ylim(min_y_value,max_y_value)
            # plt.subplots_adjust(left=0.1,
            #         bottom=0.1,
            #         right=0.9,
            #         top=0.9,
            #         wspace=0.4,
            #         hspace=0.4)
            # plt.savefig('images/DTW4/'+data +"_" +str(class_number)+'.pdf')
            # # plt.subplot_tool()
            # plt.show()

            vs = VS.Dataset_Visualization()
            vs.Plot_one(xtest[min_indexes[i][class_number]],title =data +"_" +str(class_number)+ "_min_dtw_original",save=True
                        ,ylimite=(min_y_value,max_y_value),path = 'images/DTW1/')
            vs.Plot_one(pred_df2[min_indexes[i][class_number]],title =data +"_" +str(class_number)+ "_min_dtw_predicted",save=True
                        ,ylimite=(min_y_value,max_y_value),path = 'images/DTW1/')
            vs.Plot_one(xtest[max_indexes[i][class_number]],title =data +"_" +str(class_number)+ "_max_dtw_original",save=True
                        ,ylimite=(min_y_value,max_y_value),path = 'images/DTW1/')
            vs.Plot_one(pred_df2[max_indexes[i][class_number]],title =data +"_" +str(class_number)+ "_max_dtw_predicted",save=True
                        ,ylimite=(min_y_value,max_y_value),path = 'images/DTW1/')
        # vs.Plot_one(xtest_permutated[0],title = "Permutated for "+data,save=True)
        i +=1