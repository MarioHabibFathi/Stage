#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 17:01:11 2023

@author: mariohabibfathi
"""

from Constants import UCR_PATH
import Datasets as ds
import visualization as VS
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import random
import matplotlib.pyplot as plt
import os


d = ds.Dataset(UCR_PATH)

# path_output = 'outputs/FCNEncoderDecoder/MSE-kernfilteror/1-segmentation/'
# path_old_pred = 'outputs/FCNEncoderDecoder/MSE-kernfilteror/3-segmentation/'
image_outputs_con = 'images/FCNEncoderDecoder/run1/MSE-kernfilteror/random_samples_concat/'
image_outputs_sep = 'images/FCNEncoderDecoder/run1/MSE-kernfilteror/random_samples_separated/'

datasets = ['SmoothSubspace','UMD','ECG200','Beef']
result_path = 'outputs/FCNEncoderDecoder/'
output_segment_directory = ['1-segmentation/','3-segmentation/']
all_files = os.listdir(result_path)
exclude_file = 'nn-dtw'
image_outputs_path = 'images/FCNEncoderDecoder/run1/'
image_type = ['random_samples_separated/','random_samples_concat/']

filtered_files = [f for f in all_files if f != exclude_file]


for data in datasets:
                
    xtrain,ytrain,xtest,ytest = d.load_dataset(data)
    
    xtest = d.znormalisation(xtest)
    size = xtest.shape[0]
    random_samples = random.sample(range(1, size), 3)
        
    for file in filtered_files:    
    

        segments_1 = np.load(result_path+file+'/'+output_segment_directory[0]+data+'_predicted.npy')        
        segments_3 = np.load(result_path+file+'/'+output_segment_directory[1]+data+'_predicted.npy')
        
        original_sample = xtest[random_samples]
        predection_1_seg = segments_1[random_samples]
        predection_3_seg = segments_3[random_samples]

        min_y_value = min(np.min(original_sample),np.min(predection_1_seg),
                          np.min(predection_3_seg))-0.1
        max_y_value = max(np.max(original_sample),np.max(predection_1_seg),
                          np.max(predection_3_seg))+0.1



        separation_path = image_outputs_path+image_type[0]+file
        if not os.path.exists(separation_path):
            os.makedirs(separation_path)
    
        concatination_path = image_outputs_path+image_type[1]+file
        if not os.path.exists(concatination_path):
            os.makedirs(concatination_path)



        if '-kernfilteror' in file:
            measure_metric, layers_type = file.split('-')
            layers_type = 'Modified_value'
        else:
            measure_metric = file
            layers_type = 'Default_value'



        for i in range(3):
            
            title = data+'_'+str(i)+'_'+measure_metric+'_'+layers_type
            
            fig = plt.figure()
            fig.set_figheight(9)
            fig.set_figwidth(9)
            fig.suptitle(title.replace('_',' '))
            
            ax1 = plt.subplot2grid(shape=(3, 3), loc=(0, 0)
                                    , colspan=3)
                                    # , rowspan=3)
            ax2 = plt.subplot2grid(shape=(3, 3), loc=(1, 0)
                                    , colspan=3)
                                    # , rowspan=3)
            ax3 = plt.subplot2grid(shape=(3, 3), loc=(2, 0)
                                    , colspan=3)
                                    # , rowspan=3)
            
            ax1.plot(original_sample[i], color="blue")
            ax1.set_title('Orginal')
            ax1.axis(ymin = min_y_value,ymax = max_y_value)
    
            ax2.plot(predection_1_seg[i], color="blue")
            ax2.set_title("1 segmentation")
            ax2.axis(ymin = min_y_value,ymax = max_y_value)
    
            
            ax3.plot(predection_3_seg[i], color="blue")
            ax3.set_title("3 segmentation" )
            ax3.axis(ymin = min_y_value,ymax = max_y_value)
            plt.tight_layout()
            plt.savefig(image_outputs_path+image_type[0]+file+'/'+title+'.pdf')
            plt.savefig(image_outputs_path+'All/'+image_type[0]+title+'.pdf')    
            plt.show()
        
            plt.plot(original_sample[i], color="blue", label = "Original")
            plt.plot(predection_1_seg[i], color="green", label = "1 segmentation")
            plt.plot(predection_3_seg[i], color="red", label = "3 segmentation")
            plt.legend()
            plt.ylim(ymin = min_y_value,ymax = max_y_value)
            plt.title(title.replace('_',' '))
            plt.savefig(image_outputs_path+image_type[1]+file+'/'+title+'.pdf')
            plt.savefig(image_outputs_path+'All/'+image_type[1]+title+'.pdf')    
            plt.show()
