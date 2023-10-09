#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:11:06 2023

@author: mariohabibfathi
"""

from tensorflow import keras
import numpy as np
from Constants import UCR_PATH
import DataAug as DA
import Datasets as ds
import FCNEncoder


d = ds.Dataset(UCR_PATH)

path_output = 'outputs/FCNEncoder'


datasets = ['SmoothSubspace','SyntheticControl','BeetleFly','BirdChicken',
            'Coffee','UMD','Meat','ItalyPowerDemand','Chinatown','ECG200']

L = []

for data in datasets:

    xtrain,ytrain,xtest,ytest = d.load_dataset(data)
    
    xtrain = d.znormalisation(xtrain)
    xtest = d.znormalisation(xtest)
    
    # xtrain = xtrain*4
    
    DataAug = DA.DataAugmentation()
    new_xtest = np.array(DataAug.JigSaw(xtest,generat_gap= False))
    xtest_permutated = np.array([row[0] for row in new_xtest])



    model = keras.models.load_model(path_output+'/'+data+'last_model.hdf5')
    
    # L.append(model)



