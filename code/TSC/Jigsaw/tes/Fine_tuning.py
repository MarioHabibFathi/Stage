#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 12:06:32 2023

@author: mariohabibfathi
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Set TensorFlow logging level to suppress warnings
import tensorflow as tf
import numpy as np
from Constants import UCR_PATH
import Datasets as ds
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import DataAug as DA
import time
import pandas as pd


def transfer_model(original_model,n_classes,Freeze = False):
    
    # last_conv_index = None
    # for i, layer in enumerate(original_model.layers):
    #     if isinstance(layer, tf.keras.layers.Conv1D):
    #         last_conv_index = i
    
    # if last_conv_index is None:
    #     raise ValueError('Convolutional layer not found')
    
    # # Create a new model using only the encoder layers
    # # encoder_model = tf.keras.models.Model(inputs=orginal_model.input, outputs=orginal_model.layers[last_conv_index].output)
    
    # # Find the index of the first layer after the encoder
    # first_layer_after_encoder_index = None
    # for i in range(last_conv_index + 1, len(original_model.layers)):
    #     if not isinstance(original_model.layers[i], tf.keras.layers.Conv1DTranspose):
    #         first_layer_after_encoder_index = i
    #         break
    
    
    
    
    
    
    # if first_layer_after_encoder_index is None:
    #     raise ValueError('First layer after encoder not found')
    
    
    
    last_conv_index = None

    for i, layer in enumerate(original_model.layers):
        # print(layer.name)
        if isinstance(layer, tf.keras.layers.Conv1D):
            # print('gowa   ',layer.name)
            # print(i)
            first_layer_after_encoder_index = i
            last_conv_index = i
        if isinstance(layer, tf.keras.layers.Conv1DTranspose):
            first_conv_transpose_index = i
            # print('fash5 ',i)
            break

            # if last_conv_index is not None:
    if first_conv_transpose_index is None:
        raise ValueError('Conv1DTranspose layer not found')                # break

    if last_conv_index is None:
        raise ValueError('Convolutional layer not found')
    
    
    # first_layer_after_encoder_index=11
    
    
    
    
    
    
    
    # Freeze the encoder layers
    if Freeze: 
        for layer in original_model.layers[:first_layer_after_encoder_index]:
            layer.trainable = False

    modified_model = tf.keras.models.Model(inputs=original_model.input, outputs=original_model.layers[first_layer_after_encoder_index -1].output)

    
    gap = tf.keras.layers.GlobalAveragePooling1D()(modified_model.output)

    # output_shape = tf.keras.layers.Reshape(target_shape=(n_classes,1))(gap)
    # output_layer = tf.keras.layers.Activation("softmax")(gap)

    output_layer = tf.keras.layers.Dense(n_classes, activation="softmax")(gap)

    modified_model = tf.keras.models.Model(inputs=modified_model.input,outputs=output_layer)
    
    
    my_learning_rate = 0.001
    my_optimizer = tf.keras.optimizers.Adam(learning_rate=my_learning_rate)
    
    # my_loss = triplet_loss_function(alpha=alpha)
    
    my_loss = tf.keras.losses.CategoricalCrossentropy()

    
    modified_model.compile(loss=my_loss,optimizer=my_optimizer, metrics=[ 'accuracy'])
    # tf.keras.utils.plot_model(modified_model, to_file="outputs/FCNEncoderDecoder/run4/jigsaw/resultes/elsa7/aaatest_mossssssssssssdels_for_12_to_15_permutatrion_15_model.pdf", show_shapes=True)

    return modified_model



# def transfer_model(original_model, n_classes, Freeze=False):
#     last_conv_index = None

#     for i, layer in enumerate(original_model.layers):
#         print(layer.name)
#         if isinstance(layer, tf.keras.layers.Conv1D):
#             print('gowa   ',layer.name)
#             last_conv_index = i
#         if isinstance(layer, tf.keras.layers.Conv1DTranspose):
#             first_conv_transpose_index = i
#             break

#             # if last_conv_index is not None:
#     if first_conv_transpose_index is None:
#         raise ValueError('Conv1DTranspose layer not found')                # break

#     if last_conv_index is None:
#         raise ValueError('Convolutional layer not found')

#     print(len(original_model.layers))
#     print(last_conv_index)
#     print(first_conv_transpose_index)
#     encoder_layers = original_model.layers[:last_conv_index ]
#     encoder_output = encoder_layers[-1].output

#     encoder_model = tf.keras.models.Model(inputs=original_model.input, outputs=encoder_output)

#     if Freeze:
#         for layer in encoder_layers:
#             layer.trainable = False

#     gap = tf.keras.layers.GlobalAveragePooling1D()(encoder_output)
#     output_layer = tf.keras.layers.Dense(n_classes, activation="softmax")(gap)

#     modified_model = tf.keras.models.Model(inputs=encoder_model.input, outputs=output_layer)

#     my_learning_rate = 0.001
#     my_optimizer = tf.keras.optimizers.Adam(learning_rate=my_learning_rate)
#     my_loss = tf.keras.losses.CategoricalCrossentropy()

#     modified_model.compile(loss=my_loss, optimizer=my_optimizer, metrics=['accuracy'])
#     tf.keras.utils.plot_model(encoder_model, to_file="outputs/FCNEncoderDecoder/run4/jigsaw/resultes/test_models_for_12_to_15_permutatrion_15_encoder.pdf", show_shapes=True)

#     return modified_model













def predict_fn(model, data):
    return model.predict(data)




d = ds.Dataset(UCR_PATH)

datasets = ['SmoothSubspace','UMD','ECG200','Beef']
# datasets=['UMD']


#RMSE
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)                 


# for seg_num in range(16,21):
#     print("We are at segment number ",seg_num)
#     for permutation_num in range(2,seg_num):
#         print("We are at segment number ",seg_num," and permutation number ",permutation_num)

        # path_output = path_output_first_part + str(seg_num)+continue_path_output
        
        
seg_num = 3
permutation_num = 1
     
segments_acc = []
segments_err_rate = []
segments_std = []
segments_indx = []
segments_time = []
for seg_num in range(25,26):   
    for permutation_num in range(7,seg_num+1):
        path_output = f'outputs/FCNEncoderDecoder/run4/jigsaw/{seg_num}_Segments/{permutation_num}-permutation/nobottleneck/'
        
        
        
        
        start = time.time()
        for data in datasets:
            avg_acc = []
            print(data)
            for i in range(5):
                
                xtrain,ytrain,xtest,ytest = d.load_dataset(data)
                
                xtrain = d.znormalisation(xtrain)
                xtest = d.znormalisation(xtest)
                
                # xtrain = xtrain*4
            
                
                LE = LabelEncoder()
                
                ytrain = LE.fit_transform(ytrain)
                ytest = LE.fit_transform(ytest)
                
                
                # print(len(np.unique(ytrain)))
                
                ytrain = to_categorical(ytrain)
                # ytest = to_categorical(ytest)
                # print(ytrain.shape)
                original_model = tf.keras.models.load_model(path_output+data+'_model.hdf5')
                # print(original_model.summary)
                
                
                
                
                
                DataAug = DA.DataAugmentation(segments_num=seg_num)
                
                
                
                xtrain_aug,xtrain_new = np.array(DataAug.JigSaw(xtrain,generat_gap= False,Augmented_data= False,num_of_new_copies=permutation_num))
                xtrain_permutated = xtrain_new[0]
                
                _,xtest_new = np.array(DataAug.JigSaw(xtest,generat_gap= False,Augmented_data= False))
                xtest_permutated =xtest_new[0]
            
                
                
                
                
                modified_model = transfer_model(original_model,ytrain.shape[1],Freeze=True)
                
                # X_train, X_val, y_train, y_val = train_test_split(xtrain,ytrain , test_size=0.2, random_state=42)
                X_train_permutated, X_val_permutated, y_train, y_val = train_test_split(xtrain_permutated, ytrain, test_size=0.2, random_state=42)
                
                
                min_loss = 1e9
            
                reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=5, min_lr=min_loss)
                
                # history = modified_model.fit(X_train, y_train, epochs=1000, batch_size=64, validation_data=(X_val, y_val),callbacks=reduce_lr,verbose=0)
            
                history = modified_model.fit(X_train_permutated, y_train, epochs=1000, batch_size=64, validation_data=(X_val_permutated, y_val),callbacks=reduce_lr,verbose=0)
            
                # # Get the training history
                # train_loss = history.history['loss']
                # val_loss = history.history['val_loss']
                # train_accuracy = history.history['accuracy']
                # val_accuracy = history.history['val_accuracy']
                
                # # Plot the loss curves
                # plt.plot(train_loss, label='Train Loss')
                # plt.plot(val_loss, label='Validation Loss')
                # plt.xlabel('Epochs')
                # plt.ylabel('Loss')
                # plt.legend()
                # plt.show()
                
                # # Plot the accuracy curves
                # plt.plot(train_accuracy, label='Train Accuracy')
                # plt.plot(val_accuracy, label='Validation Accuracy')
                # plt.xlabel('Epochs')
                # plt.ylabel('Accuracy')
                # plt.legend()
                # plt.show()
                
                # predictions = modified_model.predict(xtest_permutated)
                predictions = predict_fn(modified_model,xtest_permutated)
                y_pred = np.argmax(predictions, axis=1)
            
                # accuracy = np.mean(ytest == y_pred)
                accuracy = accuracy_score(ytest,y_pred)
                avg_acc.append(accuracy)
                
                print(f'we are at run {i} with accuracy of {accuracy}')
            
            
                        # fcn = FCNEncoderDecoder.FCN(X_train_permutated.shape[1],output_directory=path_output+data,
                        #               batch_size=64,epochs=1000)
            end_time = time.time() - start
            avg = np.mean(avg_acc)
            std = np.std(avg_acc)
            segments_acc.append(avg)
            segments_err_rate.append(1-avg)
            segments_std.append(std)
            segments_indx.append(f'{data} with {seg_num} segments for {permutation_num} permutations')
            segments_time.append(end_time)
            
            print(f'The average accuracy for {data} is {avg} and a std of {std} for segmentation {seg_num} and at {permutation_num} permutation for total time of {end_time} seconds')
        
        
    
    
        df = pd.DataFrame(list(zip(segments_indx,segments_acc,segments_err_rate,segments_std,segments_time)),
                          columns = ['Test Type','Accuracy','Error Rate','STD','Total Time'])
        df.to_csv('outputs/FCNEncoderDecoder/run4/jigsaw/resultes/elsa7/test_models_for_all_permutation_25.7_freeze_nobottle.csv')

    # Unable to load conversation 383a42be-3779-428d-9980-9cd828a8af76
    
    
    

    #     err_rate = []
    #     dataset_result = []
    #     dataset_std = []
    #     dataset_result_no_dia = []
    #     dataset_std_no_dia = []
    #     dataset_resultnp = []

    # # for i in range(10):

    #     print(data)

        
    #     xtrain,ytrain,xtest,ytest = d.load_dataset(data)
        
    #     xtrain = d.znormalisation(xtrain)
    #     xtest = d.znormalisation(xtest)
        
    #     # xtrain = xtrain*4
    
        
    #     LE = LabelEncoder()
        
    #     ytrain = LE.fit_transform(ytrain)
    #     ytest = LE.fit_transform(ytest)