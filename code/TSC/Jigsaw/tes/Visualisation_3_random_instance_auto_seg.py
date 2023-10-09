#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 12:06:38 2023

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

# d.Show_Dataset()
# path_output = 'outputs/FCNEncoder_multipledata/'
# result_path = 'outputs/FCNEncoder_multipledata/DTW_resultes/'
# path_old_pred = 'outputs/FCNEncoder/'
# image_outputs = 'images/multipledata/run2/random_samples_concat/'
# image_outputs_sep = 'images/multipledata/run2/random_samples_separated/'

path_data = 'outputs/FCNEncoderDecoder/run3/jigsaw/'

# path_old_pred = 'outputs/FCNEncoderDecoder/run3/jigsaw/'
# path_output = 'outputs/FCNEncoderDecoder/run3/jigsaw/'
# result_path = 'outputs/FCNEncoderDecoder/1-segmentation/DTW_resultes/'


path_image = 'images/FCNEncoderDecoder/run3/'
# image_outputs_con = 'images/FCNEncoderDecoder/run3/'
# image_outputs_sep = 'images/FCNEncoderDecoder/run3/'


Bottleneck = ['_Segments/1-permutation/nobottleneck/','_Segments/1-permutation/bottleneck/']
image_path = ['_Segments/bottlenecktest/random_samples_concat/','_Segments/bottlenecktest/random_samples_separated/']
# df = pd.read_csv(result_path+'DTW_resultes.csv',index_col=0)
# ,usecols=['Dataset','min index','max index'])


# datasets = df.iloc[:,0]
# min_index = df.iloc[:,1]
# max_index = df.iloc[:,2]

# i = 0
datasets = ['SmoothSubspace','UMD','ECG200','Beef']
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)                 



for seg_num in range(21,41):
    
    
    
    path_old_pred = path_data+str(seg_num)+Bottleneck[0]
    path_output = path_data+str(seg_num)+Bottleneck[1]
    image_outputs_con = path_image +str(seg_num)+ image_path[0]
    image_outputs_sep = path_image +str(seg_num)+ image_path[1]

    paths = [path_old_pred,path_output,image_outputs_con,image_outputs_sep]
    for p in paths:
        
        if not os.path.exists(p):
            os.makedirs(p)


    
    for data in datasets:
            
        # for i in range(10):
            
            xtrain,ytrain,xtest,ytest = d.load_dataset(data)
            
            # xtrain = d.znormalisation(xtrain)
            xtest = d.znormalisation(xtest)
            size = xtest.shape[0]
            random_samples = random.sample(range(1, size), 3)
            
            
            # xtrain = xtrain*4
        
            
            # LE = LabelEncoder()
            
            # ytrain = LE.fit_transform(ytrain)
            # ytest = LE.fit_transform(ytest)
            # pred_df1 = np.load(path_old_pred+data+'_predicted.npy')        
            # pred_df2 = np.load(path_output+data+'_predicted.npy')
            
            # original_sample = xtest[random_samples]
            # prediction1 = pred_df1[random_samples]
            # prediction2 = pred_df2[random_samples]
    
    
            segments_1 = np.load(path_old_pred+data+'_predicted.npy')        
            segments_3 = np.load(path_output+data+'_predicted.npy')
            
            original_sample = xtest[random_samples]
            predection_1_seg = segments_1[random_samples]
            predection_3_seg = segments_3[random_samples]
    
            min_y_value = min(np.min(original_sample),np.min(predection_1_seg),
                              np.min(predection_3_seg))-0.1
            max_y_value = max(np.max(original_sample),np.max(predection_1_seg),
                              np.max(predection_3_seg))+0.1
    
    
            # plt.subplot(3, 1, 1)
            # plt.plot(xtest[1], color="blue")
            # plt.title(data +" orginal" )
            # plt.title("min dtw original" )
    
            # plt.ylim(np.min(xtrain),np.max(xtrain))
    
            # plt.subplot(3,1, 2)
            # plt.plot(pred_df1[1], color="blue")
            # plt.title(data +" Old prediction")
            # plt.title("min dtw predicted" )
    
            # plt.ylim(np.min(xtrain),np.max(xtrain))
    
            # plt.subplot(3, 1, 3)
            # plt.plot(pred_df2[2], color="blue")
            # plt.title(data +" New prediction" )
            # plt.title("max dtw original" )
            
            # plt.ylim(np.min(xtrain),np.max(xtrain))
            # plt.subplots_adjust(left=0.1,
            #         bottom=0.1,
            #         right=0.9,
            #         top=0.9,
            #         wspace=0.4,
            #         hspace=0.4)
            # plt.savefig('images/DTW4/'+data +"_" +str(class_number)+'.pdf')
            # plt.subplot_tool()
            
    
            for i in range(3):
                
                fig = plt.figure()
                fig.set_figheight(9)
                fig.set_figwidth(9)
                fig.suptitle(data)
                
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
                ax2.set_title("no bottleneck")
                # ax2.axis(np.min(xtrain),np.max(xtrain))
                ax2.axis(ymin = min_y_value,ymax = max_y_value)
        
                
                ax3.plot(predection_3_seg[i], color="blue")
                ax3.set_title("with bottleneck" )
                # # ax3.axis(np.min(xtrain),np.max(xtrain))
                # ax3.axis(ymin = min_y_value,ymax = max_y_value)
                plt.tight_layout()
                plt.savefig(image_outputs_sep+data+str(i)+'.pdf')
        
                plt.show()
            
                plt.plot(original_sample[i], color="blue", label = "Original")
                plt.plot(predection_1_seg[i], color="green", label = "no bottleneck")
                plt.plot(predection_3_seg[i], color="red", label = "with bottleneck")
                plt.legend()
                plt.ylim(ymin = min_y_value,ymax = max_y_value)
                plt.title(data)
                plt.savefig(image_outputs_con+data+str(i)+'.pdf')
    
                plt.show()
    
            
            
         