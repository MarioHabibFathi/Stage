#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 10:58:14 2023

@author: mariohabibfathi
"""

# import Constants
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
import os

d = ds.Dataset(UCR_PATH)

# d.Show_Dataset()
path_output_first_part = 'outputs/FCNEncoderDecoder/run4/jigsaw/'
continue_path_output = '_Segments/1-permutation/nobottleneck/'
AUDA =  False
# seg_num = 5

path_images_output = 'images/multipledata/run2/main/'
save_image = False

datasets = ['SmoothSubspace','SyntheticControl','BeetleFly','BirdChicken',
            'Coffee','UMD','Meat','ItalyPowerDemand','Chinatown','ECG200']

# datasets = ['BeetleFly']
# datasets = ['Meat','ItalyPowerDemand','Chinatown','ECG200']
    
datasets = ['DistalPhalanxTW','MiddlePhalanxTW','Fungi','Beef','CBF']
datasets = [ 'Plane','ArrowHead','Trace','OliveOil','Ham','Herring','InsectWingbeatSound','Lightning7',
              'ECGFiveDays','Lightning2','Adiac']
# 
datasets = ['Wafer','SwedishLeaf','TwoLeadECG']

# datasets = ['ShakeGestureWiimoteZ','PickupGestureWiimoteZ','GesturePebbleZ2','GesturePebbleZ1'
#             ,'Rock']


datasets = ['SmoothSubspace','UMD','ECG200','Beef']
# datasets=['UMD']

acc = []
acc = []
err_rate = []
num_runs = 5
num_ker = 10_000

#RMSE
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)                 


for seg_num in range(21,41):
    path_output = path_output_first_part + str(seg_num)+continue_path_output

    if not os.path.exists(path_output):
        os.makedirs(path_output)
    print(path_output)
    
    for data in datasets:
            err_rate = []
            dataset_result = []
            dataset_std = []
            dataset_result_no_dia = []
            dataset_std_no_dia = []
            dataset_resultnp = []
    
        # for i in range(10):
    
            print(data)
    
            
            xtrain,ytrain,xtest,ytest = d.load_dataset(data)
            
            xtrain = d.znormalisation(xtrain)
            xtest = d.znormalisation(xtest)
            
            # xtrain = xtrain*4
        
            
            LE = LabelEncoder()
            
            ytrain = LE.fit_transform(ytrain)
            ytest = LE.fit_transform(ytest)
            
            # ytrain = ytrain*4
            
            # seg_num = int(xtrain.shape[1]/10)
            # print("sssssssssssssssssss ",seg_num)
            # print("sssssssssssssssssss ",seg_num)
            # print("sssssssssssssssssss ",seg_num)
            # print("sssssssssssssssssss ",seg_num)
            # print("sssssssssssssssssss ",seg_num)
            # print("sssssssssssssssssss ",seg_num)
            # print("sssssssssssssssssss ",seg_num)
            # print("sssssssssssssssssss ",seg_num)
            # seg_num = 4
            # d
            # d
            # d
            # d
            # dd
            # d
            DataAug = DA.DataAugmentation(segments_num=seg_num)
            
            
            
            
    #         zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
            
            xtrain_aug,xtrain_new = np.array(DataAug.JigSaw(xtrain,generat_gap= False,Augmented_data= AUDA))
            # xtrain_permutated = np.array([row[0] for row in xtrain_new])
            xtrain_permutated = xtrain_new[0]
            # xtrain_permutated = np.expand_dims(xtrain_permutated, axis=1)
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
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
    
    
            # d
            # d
            # d
            # d
            # d
            # d
            # DataAug = DA.DataAugmentation(segments_num=seg_num)
            _,xtest_new = np.array(DataAug.JigSaw(xtest,generat_gap= False,Augmented_data= False))
            # xtest_permutated = np.array([row[0] for row in new_xtest])
            xtest_permutated =xtest_new[0]
    
            # xtest_permutated=np.expand_dims(xtest_permutated, axis=1)
    
            # aaa = np.array(new_xtrain[:,0],dtype = 'float64')
            # aaa2 = np.array([row[1] for row in new_xtrain])
    
            # DataAug._generat_gap_list(8)
            # print(len(min(xtrain, key=len)))
            # print(7//2)
            # aaa = da.Devide_ser(xtrain[0])
            # aaa = da.Permutation(aaa)
            # a = da.merge(aaa)
            # aaa = DataAug.Devide_ser(xtrain[0])
            # aaa = DataAug.Permutation(aaa)
            # a = DataAug.merge(aaa)
            # a = da.merge(aaa)
            # da.Permutation(a)
            # a = DataAug.generat_data(xtrain[0])
            # print(aaa[0])
            
            # vs = VS.Dataset_Visualization()
            # vs.Plot_one(xtrain[0])
            # vs.Plot_one(a)
            # vs.Plot_one(aaa)
            
            # ytrain = to_categorical(ytrain)
            # ytest = to_categorical(ytest)
            
            
            # dddddddddddddddd
            
            
            
            X_train, X_val, _, _ = train_test_split(xtrain_aug,xtrain_aug , test_size=0.2, random_state=42)
            X_train_permutated, X_val_permutated, _, _ = train_test_split(xtrain_permutated, xtrain_permutated, test_size=0.2, random_state=42)
            
            
            # X_train, X_val, _, _ = train_test_split(xtrain,xtrain , test_size=0.2, random_state=42)
            # X_train_permutated, X_val_permutated, _, _ = train_test_split(xtrain_permutated, xtrain_permutated, test_size=0.2, random_state=42)
            
            
            
            
            # zzzzzzzzzzzzz
            # fcn = FCN.FCN(len(X_train[0][0]),n_classes=ytrain.shape[1],output_directory=path_output+'/'+data,
            #               batch_size=min(int(len(X_train[0][0])/10), 16),epochs=1000)
            
            # fcn = FCN.FCN(X_train.shape[1],n_classes=ytrain.shape[1],output_directory=path_output+'/'+data,
            #               batch_size=min(int(X_train.shape[0]/10), 16),epochs=1000)
    
            # print(xtrain_permutated.shape)
    
    # min(int(X_train_permutated.shape[0]/10), 16)
    
    
            fcn = FCNEncoderDecoder.FCN(X_train_permutated.shape[1],output_directory=path_output+data,
                          batch_size=64,epochs=1000)
            
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
           
            
           
            
           
            # zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
            # a12 = tf.argmax(pred,axis=1)
            # a11 = to_categorical(a12)
            # dataset_result.append(accuracy_score(a12, ytest))
            # err_rate.append(1-accuracy_score(a12, ytest))
        
            # # # print(np.unique(ytrain))
            
            # # model = LogisticRegression(solver='sag', max_iter=150)
            # # model.fit(xtrain, ytrain)
            # # pred = model.predict(xtest)
            
            # lr = LR.LogisticRegression(regularization='l1')
            # lr.fit(xtrain, ytrain)
            # pred = lr.predict(xtest)
            # print(accuracy_score(ytest, pred))
            
            # # xxxxxxxxxxxxxxxx
            # # sk = svm.SVC()
        
            # # sk.fit(xtrain, ytrain)
            # # pred = sk.predict(xtest)
            # # SVM = CL.SVM()
            
            # # pred = NN.k_nearest_neighbor(xtrain, ytrain, xtest, metric_distance='DTW')
            
            # # SVM.fit(xtrain, ytrain)
        
            # # pred = SVM.predict(xtest)
        
            # acc.append(accuracy_score(ytest, pred))
            # err_rate.append(1-accuracy_score(ytest, pred))
        
            # # print("dsds")
            # print("run",i)
    
            # print("Done "+data)
            # result = pd.DataFrame(list(zip(dataset_result,err_rate)),
            #                         columns = ["accuracy","Error rate"
            #                                   ])
            # result.to_csv('outputs/10run/UCR_1_examples_FCN_predection_Nogap_test_'+data+'_10_run.csv')


# print(1.0 - accuracy_score(ytest, pred))
# df = pd.DataFrame(list(zip(datasets,acc,err_rate)),columns = ['Dataset','accuracy','Error rate'])
# df.to_csv('outputs/LR/Test me/result_ovr_no_reg.csv')

# VS.Dataset_Visualization().Plot_all(xtrain,ytrain,save=True)
# VS.Dataset_Visualization().Plot_all_by_Class(xtrain,ytrain,save=True)

# VS.Dataset_Visualization().Plot_by_Class(xtrain,ytrain,save=True)

# VS.Dataset_Visualization().Plot_individual(xtrain,ytrain,save=True)