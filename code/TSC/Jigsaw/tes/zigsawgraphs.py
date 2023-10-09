#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 17:14:54 2023

@author: mariohabibfathi
"""

from Constants import UCR_PATH
import Datasets as ds
import DataAug as DA
import numpy as np
import matplotlib.pyplot as plt


d = ds.Dataset(UCR_PATH)
datasets = ['SmoothSubspace','UMD','ECG200','Beef']

datasets = ['SmoothSubspace']
path_image_data = 'images/FCNEncoderDecoder/run4/jigsaw/accuracy/'

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
        
        DataAug = DA.DataAugmentation(segments_num=3)

        xtrain_aug,xtrain_new = np.array(DataAug.JigSaw(xtrain,generat_gap= True,Augmented_data= True,num_of_new_copies=3))
        # xtrain_permutated = np.array([row[0] for row in xtrain_new])
        xtrain_permutated = xtrain_new[0]

        plt.plot(xtrain_aug[0], label = "Original")
        plt.plot(xtrain_permutated[0], label = "1st permutation")
        print(xtrain_aug[0])
        print(xtrain_permutated[0])
        # plt.plot(xtrain_permutated[1], label = "2nd permutation")
        # plt.plot(xtrain_permutated[2], label = "3rd permutation")
        # plt.plot(filtered_data_NEW['Accuracy No Bottleneck No Freeze'], label = "No Bottleneck No Freeze New")
        # plt.plot(filtered_data_NEW['Accuracy No Bottleneck Freeze'], label = "No Bottleneck Freeze New")
        plt.title(f'Different permuatation for the same series for {data}')
        # plt.ylim(0,1)
        plt.legend(fontsize="9")
        # plt.xlabel('Number of segments')
        # plt.ylabel('Accuracy')
        # plt.savefig(path_image_data+'pdf/'+f'{data}_permutation.pdf')
        # plt.savefig(path_image_data+'png/'+f'{data}_permutation.png')