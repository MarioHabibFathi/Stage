#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 10:40:28 2023

@author: mariohabibfathi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
path_image_data = 'images/FCNEncoderDecoder/run4/jigsaw/accuracy/'

datasets = ['SmoothSubspace','UMD','ECG200','Beef']
df = pd.read_csv('outputs/FCNEncoderDecoder/run4/jigsaw/resultes/Send/Fine_tunning_1-permutation_resultes_Simplified.csv')
df2 = pd.read_csv('outputs/FCNEncoderDecoder/run4/jigsaw/resultes/The last resultes/Fine_tunning_40-segments_resultes_Simplified.csv')
# df_New = pd.read_csv('outputs/FCNEncoderDecoder/run4/jigsaw/resultes/elsa7/Comaparison_test_models_for_1_permutation_2_to_40_nobottle_Simplified.csv')
threshold_values = [0.97,0.9875,0.89,0.68]
i=0
# datasets = ['SmoothSubspace']
for datas in datasets:
    filtered_data = df[df['Test Type'].str.contains(datas)].reset_index(drop=True)
    filtered_data.index = np.arange(2, len(filtered_data) + 2)
    filtered_data2 = df2[df2['Test Type'].str.contains(datas)].reset_index(drop=True)
    filtered_data2.index = np.arange(2, len(filtered_data2) + 2)
    # filtered_data_NEW = df_New[df_New['Test Type'].str.contains(datas)].reset_index(drop=True)
    # filtered_data_NEW.index = np.arange(2, len(filtered_data_NEW) + 2)
    # filtered_data = filtered_data[filtered_data['Test Type'].str.contains('Beef')]
    
    # plt.plot(filtered_data['Accuracy No Bottleneck No Freeze'], label = "No Bottleneck No Freeze segm")
    # plt.plot(filtered_data['Accuracy No Bottleneck Freeze'], label = "No Bottleneck Freeze segm")
    # plt.plot(filtered_data['Accuracy Bottleneck No Freeze'], label = "Bottleneck No Freeze")
    # plt.plot(filtered_data['Accuracy Bottleneck Freeze'], label = "Bottleneck Freeze")
    plt.plot(filtered_data2['Accuracy No Bottleneck No Freeze'], label = "No Bottleneck No Freeze")
    plt.plot(filtered_data2['Accuracy No Bottleneck Freeze'], label = "No Bottleneck Freeze")
    # plt.plot(filtered_data_NEW['Accuracy No Bottleneck No Freeze'], label = "No Bottleneck No Freeze New")
    # plt.plot(filtered_data_NEW['Accuracy No Bottleneck Freeze'], label = "No Bottleneck Freeze New")
    # threshold = float(threshold_values[i])  # Replace with your desired threshold value
    threshold = threshold_values[i]
    # plt.plot(threshold, label = "Bottleneck Freeze")
    # plt.plot([filtered_data.index[0], filtered_data.index[-1]], [threshold, threshold], color='r', linestyle='--', label='Threshold')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Reference value')
    plt.text(filtered_data.index[-1] + 3, threshold, str(threshold), color='r')
    # plt.title(f'{datas} for 1 permutation and increasing segmentation number')
    plt.title(f'{datas} for 40 segmentations \n and increasing permutation number')
    plt.ylim(0,1)
    plt.legend()
    plt.xlabel('Number of permutations')
    plt.ylabel('Accuracy')
    # plt.savefig(path_image_data+'pdf/'+f'{datas}_accuracy_for_40_segmentation_ref.pdf')
    # plt.savefig(path_image_data+'png/'+f'{datas}_accuracy_for_40_segmentation_ref.png')

    # plt.xlim(0,120)
    plt.show()
    i+=1