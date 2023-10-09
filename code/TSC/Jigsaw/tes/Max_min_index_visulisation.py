#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 10:07:20 2023

@author: mariohabibfathi
"""

from Constants import UCR_PATH
import Datasets as ds
import visualization as VS
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


d = ds.Dataset(UCR_PATH)

# d.Show_Dataset()
path_output = 'outputs/FCNEncoder_multipledata'

df = pd.read_csv('outputs/FCNEncoder/DTW_resultes.csv',index_col=0)
# ,usecols=['Dataset','min index','max index'])



datasets = df.iloc[:,0]
min_index = df.iloc[:,1]
max_index = df.iloc[:,2]

i = 0


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
        
        min_y_value = min(min(xtest[min_index[i]]),min(pred_df2[min_index[i]]),
                          min(xtest[max_index[i]]),min(pred_df2[max_index[i]]))-0.1
        max_y_value = max(max(xtest[min_index[i]]),max(pred_df2[min_index[i]]),
                          max(xtest[max_index[i]]),max(pred_df2[max_index[i]]))+0.1
        
        
        
        vs = VS.Dataset_Visualization()
        vs.Plot_one(xtest[min_index[i]],title = data + " original min DTW",save=True
                    ,ylimite=(min_y_value,max_y_value),path = 'images/multipledata/1/')
        vs.Plot_one(pred_df2[min_index[i]],title = data+" predicted min DTW",save=True
                    ,ylimite=(min_y_value,max_y_value),path = 'images/multipledata/1/')
        vs.Plot_one(xtest[max_index[i]],title = data+" original max DTW",save=True
                    ,ylimite=(min_y_value,max_y_value),path = 'images/multipledata/1/')
        vs.Plot_one(pred_df2[max_index[i]],title = data+" predicted max DTW",save=True
                    ,ylimite=(min_y_value,max_y_value),path = 'images/multipledata/1/')
        # vs.Plot_one(xtest_permutated[0],title = "Permutated for "+data,save=True)
        i +=1







