#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 10:30:02 2023

@author: mariohabibfathi
"""

from Constants import UCR_PATH
import numpy as np
import pandas as pd
from tslearn.metrics import dtw
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
import Datasets as ds
from sklearn.preprocessing import LabelEncoder
import numpy as np
import DataAug as DA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

d = ds.Dataset(UCR_PATH)



# path_old_pred = 'outputs/FCNEncoder/'
# path_old_pred = 'outputs/FCNEncoder_multipledata/'
path_old_pred = 'outputs/FCNEncoderDecoder/MAE/3-segmentation/'


output_path_dataframe = 'outputs/FCNEncoderDecoder/nn-dtw/'
datasets = ['SmoothSubspace','SyntheticControl','BeetleFly','BirdChicken',
            'Coffee','UMD','Meat','ItalyPowerDemand','Chinatown','ECG200']
datasets = ['SmoothSubspace','SyntheticControl','BeetleFly','BirdChicken',
            'Coffee','UMD','Meat','ItalyPowerDemand','Chinatown','ECG200'
            ,'DistalPhalanxTW','MiddlePhalanxTW','Fungi','Beef','CBF',
            'Plane','ArrowHead','Trace','OliveOil','Ham','Herring',
            'InsectWingbeatSound','Lightning7','ECGFiveDays','Lightning2','Adiac',            
            'Wafer','SwedishLeaf','TwoLeadECG','ShakeGestureWiimoteZ','PickupGestureWiimoteZ',
            'GesturePebbleZ2','GesturePebbleZ1','Rock']

datasets = ['SmoothSubspace','SyntheticControl','BeetleFly','BirdChicken',
            'Coffee','UMD','Meat','ItalyPowerDemand','Chinatown','ECG200'
            ,'DistalPhalanxTW','MiddlePhalanxTW','Fungi','Beef','CBF',
            'Plane','ArrowHead','Trace','OliveOil','Ham','Herring',
            'InsectWingbeatSound','Lightning7','ECGFiveDays','Lightning2','Adiac',            
            'SwedishLeaf','TwoLeadECG','ShakeGestureWiimoteZ','PickupGestureWiimoteZ',
            'GesturePebbleZ2','GesturePebbleZ1','Rock']

datasets = ['SmoothSubspace','UMD','ECG200','Beef']

# datasets = ['SmoothSubspace']
# datasets = ['SwedishLeaf','TwoLeadECG','ShakeGestureWiimoteZ','PickupGestureWiimoteZ',
 # 'GesturePebbleZ2','GesturePebbleZ1','Rock']
# acc = []
# err_rate = []
num_runs = 5
num_ker = 10_000
err_rate_concatenated = []
dataset_result_concatenated = []
err_rate_enc_data = []
dataset_result_enc_data = []
# dataset_std = []
# dataset_result_no_dia = []
# dataset_std_no_dia = []
# dataset_resultnp = []

for data in datasets:

    # for i in range(10):



        
        xtrain,ytrain,xtest,ytest = d.load_dataset(data)
        
        xtrain = d.znormalisation(xtrain)
        xtest = d.znormalisation(xtest)
        
        # xtrain = xtrain*4
    
        
        LE = LabelEncoder()
        
        ytrain = LE.fit_transform(ytrain)
        ytest = LE.fit_transform(ytest)
        
        ytest_all = np.concatenate((ytest, ytest))
        
        
        predicted_encoder_data = np.load(path_old_pred+data+'_predicted.npy')        

        test_all = np.concatenate((xtest,predicted_encoder_data))
        
        knn = KNeighborsTimeSeriesClassifier(metric='dtw',n_neighbors=1)
        knn.fit(xtrain, ytrain)
        
        pred_all = knn.predict(test_all)
        acc_all = accuracy_score(ytest_all, pred_all)
        dataset_result_concatenated.append(acc_all)
        err_rate_concatenated.append(1-acc_all)
        
        
        pred_encoder_only = knn.predict(predicted_encoder_data)
        acc_enco = accuracy_score(ytest, pred_encoder_only)
        dataset_result_enc_data.append(acc_enco)
        err_rate_enc_data.append(1-acc_enco)



        print(f"Accuracy for {data} using concatenated test is {acc_all} and using encoder data is {acc_enco}")

        print(f"Error rate for {data} using concatenated test is {1-acc_all} and using encoder data is {1-acc_enco}")
        print()

df = pd.DataFrame(list(zip(datasets,dataset_result_concatenated,err_rate_concatenated,
                           dataset_result_enc_data,err_rate_enc_data)),
                  columns = ['Dataset','concatination accuracy','concatination Error rate',
                             'encoder data accuracy','encoder data Error rate']
                  )
df.to_csv(output_path_dataframe+'resulte_4_dataset_MAE_3seg.csv')




