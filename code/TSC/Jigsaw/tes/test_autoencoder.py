#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 15:05:00 2023

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

# from sklearn import svm
from sklearn.linear_model import LogisticRegression
import numpy as np
import DataAug as DA
import FCN
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import FCNEncoder
import FCNEncoderDecoder
import FCNEncoderDecoderActivation
import os
import time


d = ds.Dataset(UCR_PATH)

# d.Show_Dataset()
path_output_decoder = 'outputs/FCNEncoderDecoder/'
# AUDA = False

output_directory = ['1-segmentation/','3-segmentation/']

path_images_output = 'images/multipledata/run2/main/'
save_image = False

# datasets = ['SmoothSubspace','SyntheticControl','BeetleFly','BirdChicken',
#             'Coffee','UMD','Meat','ItalyPowerDemand','Chinatown','ECG200']

# datasets = ['BeetleFly']
# datasets = ['Meat','ItalyPowerDemand','Chinatown','ECG200']
    
# datasets = ['DistalPhalanxTW','MiddlePhalanxTW','Fungi','Beef','CBF']
# datasets = [ 'Plane','ArrowHead','Trace','OliveOil','Ham','Herring','InsectWingbeatSound','Lightning7',
#               'ECGFiveDays','Lightning2','Adiac']

# datasets = ['Wafer','SwedishLeaf','TwoLeadECG']

# datasets = ['ShakeGestureWiimoteZ','PickupGestureWiimoteZ','GesturePebbleZ2','GesturePebbleZ1'
#             ,'Rock']

Acivations = ['linear','relu','tanh','sigmoid','elu','swish']
datasets = ['SmoothSubspace','UMD','ECG200','Beef']

datasets=['SmoothSubspace']



all_files = os.listdir(path_output_decoder+'NoActivastion')
exclude_file = 'nn-dtw'
image_outputs_path = 'images/FCNEncoderDecoder/run1/'
image_type = ['random_samples_separated/','random_samples_concat/']

filtered_files = [f for f in all_files if f != exclude_file]





loss_metrics = [loss_metric for loss_metric in filtered_files if '-kernfilteror' not in loss_metric]







# acc = []
# acc = []
# err_rate = []
# num_runs = 5
# num_ker = 10_000

# loss_metrics = ['smape']
# Acivations = ['linear']


start_time = time.time()



for act in Acivations:
    for loss in loss_metrics:
        for seg in output_directory:
            # if seg == '1-segmentation/' :
            #     AUDA = False
            # else:
            #     AUDA = True
            AUDA = seg != '1-segmentation/'
            for data in datasets:
                inside_time_start = time.time()

                path_output = path_output_decoder+act+'/'+loss+'/'+seg
                
                # print(path_output)
                
                if not os.path.exists(path_output):
                    os.makedirs(path_output)
                    # err_rate = []
                    # dataset_result = []
                    # dataset_std = []
                    # dataset_result_no_dia = []
                    # dataset_std_no_dia = []
                    # dataset_resultnp = []
            
                # for i in range(10):
            
            
                # zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
            
                    
                xtrain,ytrain,xtest,ytest = d.load_dataset(data)
                
                xtrain = d.znormalisation(xtrain)
                xtest = d.znormalisation(xtest)
                
                # xtrain = xtrain*4
            
                
                LE = LabelEncoder()
                
                ytrain = LE.fit_transform(ytrain)
                ytest = LE.fit_transform(ytest)
                
                # ytrain = ytrain*4
                
                DataAug = DA.DataAugmentation()
                xtrain_aug,xtrain_new = np.array(DataAug.JigSaw(xtrain,generat_gap= False,Augmented_data= AUDA))
                # xtrain_permutated = np.array([row[0] for row in xtrain_new])
                xtrain_permutated = xtrain_new[0]
                # xtrain_permutated = np.expand_dims(xtrain_permutated, axis=1)
        
                # vs = VS.Dataset_Visualization()
                # vs.Plot_one(xtrain[0],title = "Original for "+data,save=False,path = 'images/run3/')
                # vs.Plot_one(xtrain_permutated[0][0],title = "per 1 "+data,save=False,path = 'images/run3/')
                # vs.Plot_one(xtrain_permutated[0][1],title = "per 2 "+data,save=False,path = 'images/run3/')
                # vs.Plot_one(xtrain_permutated[0][2],title = "per 3 "+data,save=False,path = 'images/run3/')
        
                # totale_size = np.prod(xtrain_permutated.shape[:2])
        
                # flat = xtrain_permutated.reshape(totale_size, -1)
                # new_shape = (totale_size*flat.shape[-1],)
                
                # xtrain_permutated_new = flat.reshape(new_shape)
                # print(xtrain_permutated.shape[0])
                # print(xtrain_permutated.shape[1])
                # print(xtrain_permutated.shape[2])
                # print(xtrain_permutated.shape[:3])
        
                # new_shape = (np.prod(xtrain_permutated.shape[:-1]),xtrain_permutated.shape[-1])
                # xtrain_permutated_new = xtrain_permutated.reshape(new_shape)
                
                
                
                
        
                # vs.Plot_one(xtrain[0],title = "Original for "+data,save=False,path = 'images/run3/')
                # vs.Plot_one(xtrain_permutated[0][0],title = "per 1 "+data,save=False,path = 'images/run3/')
                # vs.Plot_one(xtrain_permutated_new[0],title = "after per 1 "+data,save=False,path = 'images/run3/')
                # vs.Plot_one(xtrain_permutated[0][1],title = "per 2 "+data,save=False,path = 'images/run3/')
                # vs.Plot_one(xtrain_permutated_new[1],title = "after per 2 "+data,save=False,path = 'images/run3/')
                # vs.Plot_one(xtrain_permutated[0][2],title = "per 3 "+data,save=False,path = 'images/run3/')
                # vs.Plot_one(xtrain_permutated_new[2],title = "after per 3 "+data,save=False,path = 'images/run3/')
        
        
        
                DataAug = DA.DataAugmentation()
                _,xtest_new = np.array(DataAug.JigSaw(xtest,generat_gap= False,Augmented_data= True))
                # xtest_permutated = np.array([row[0] for row in new_xtest])
                xtest_permutated =xtest_new[0]
        
                
                
                X_train, X_val, _, _ = train_test_split(xtrain_aug,xtrain_aug , test_size=0.2, random_state=42)
                X_train_permutated, X_val_permutated, _, _ = train_test_split(xtrain_permutated, xtrain_permutated, test_size=0.2, random_state=42)
                # zzzzzzzzzzzzz
                # fcn = FCN.FCN(len(X_train[0][0]),n_classes=ytrain.shape[1],output_directory=path_output+'/'+data,
                #               batch_size=min(int(len(X_train[0][0])/10), 16),epochs=1000)
                
                # fcn = FCN.FCN(X_train.shape[1],n_classes=ytrain.shape[1],output_directory=path_output+'/'+data,
                #               batch_size=min(int(X_train.shape[0]/10), 16),epochs=1000)
        
                # print(xtrain_permutated.shape)
        
                fcn = FCNEncoderDecoderActivation.FCN(X_train_permutated.shape[1],output_directory=path_output+data,
                                                      batch_size=min(int(X_train_permutated.shape[0]/10), 16),
                                                      epochs=1000,activation=act,loss_metric=loss)
                
                fcn.fit(X_train_permutated, X_val_permutated,X_train, X_val)
                # fcn.fit(xtrain_permutated, xtrain, xtest_permutated, xtest)
                # fcn.fit(xtrain_permutated, xtest_permutated, xtrain, xtest)
        
                pred = fcn.predict(xtest_permutated)
               
                vs = VS.Dataset_Visualization()
                vs.Plot_one(xtest[0],title = "Original for "+data,save=save_image,path = path_images_output)
                vs.Plot_one(pred[0],title = "predicted for "+data,save=save_image,path = path_images_output)
                vs.Plot_one(xtest_permutated[0],title = "Permutated for "+data,save=save_image,path = path_images_output)
                # vs.Plot_one(aaa)
               
                
           
                # pred_array = np.asarray(pred)
                np.save(arr=pred,file=path_output+data+'_predicted.npy')
                np.save(arr=xtest_permutated,file=path_output+data+'_permutated.npy')

                print(f'Done {seg} for {data} with loss {loss} and activation {act}')
                
                inside_time_end = time.time()
                print('Total Execution time:',inside_time_end - inside_time_start , 'seconds')


end_time = time.time()
print('Total Execution time:',end_time - start_time , 'seconds')
