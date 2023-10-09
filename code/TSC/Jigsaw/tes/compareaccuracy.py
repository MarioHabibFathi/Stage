#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 12:20:20 2023

@author: mariohabibfathi
"""

import pandas as pd
import matplotlib.pyplot as plt

output_path_dataframe = 'outputs/FCNEncoder/nn-dtw/'


single_variation = pd.read_csv(output_path_dataframe+'resulte_33_dataset.csv')
multiple_variation = pd.read_csv(output_path_dataframe+'resulte_33_dataset_multipleway.csv')
ucr = pd.read_csv(output_path_dataframe+'DataSummary.csv')


dataset = single_variation['Dataset']
sin_var_con = single_variation['concatination accuracy']
sin_var_enc = single_variation['encoder data accuracy']

ucr_accuracy = {'SmoothSubspace':0.0533,'SyntheticControl': 0.0167,
                'BeetleFly': 0.3000,'BirdChicken': 0.4500,'Coffee': 0,'UMD':0.0278,
                'Meat':0.0667,'ItalyPowerDemand': 0.0447,'Chinatown': 0.0466,'ECG200': 0.1200,
                'DistalPhalanxTW': 0.3669,'MiddlePhalanxTW':0.4935,'Fungi':0.1774,'Beef': 0.3333,
                'CBF': 0.0044,'Plane': 0.0000,'ArrowHead': 0.2000,'Trace':0.0100,
                'OliveOil': 0.1333,'Ham': 0.4000,'Herring': 0.4688,'InsectWingbeatSound':0.4152,
                'Lightning7':0.4247,'ECGFiveDays': 0.2033,'Lightning2': 0.1311,'Adiac': 0.3913,
                'SwedishLeaf': 0.1536,'TwoLeadECG':0.1317,'ShakeGestureWiimoteZ':0.1600,
                'PickupGestureWiimoteZ': 0.3400,'GesturePebbleZ2': 0.2215,'GesturePebbleZ1': 0.1744,
                'Rock': 0.1600}


# da = ucr[ucr['Name'].isin(dataset)]
# aaa = pd.merge(dataset, da,left_on=dataset,right_on=da['DTW (learned_w) '])
# aaa_sor = aaa.sort_values(by=dataset)


for i in range(len(sin_var_con)):
    if sin_var_con[i] > sin_var_enc[i]:
        plt.scatter(sin_var_con[i], sin_var_enc[i],c='green',marker='o')
    elif sin_var_con[i] < sin_var_enc[i]:
        plt.scatter(sin_var_con[i], sin_var_enc[i],c='red',marker='x')
        print(dataset[i])
    else:
        plt.scatter(sin_var_con[i], sin_var_enc[i],c='blue',marker='s')

plt.xlabel("single variation concatenated")
plt.ylabel("single variation encoder")
# plt.plot([0,1],[0,1],'--',transform=plt.gca().transAxes)
plt.plot([0,1],[0,1],'--')
# plt.legend(['Tie','Win','Loss'])
plt.show()



mul_var_con = multiple_variation['concatination accuracy']
mul_var_enc = multiple_variation['encoder data accuracy']


for i in range(len(sin_var_con)):
    if mul_var_con[i] > mul_var_enc[i]:
        plt.scatter(mul_var_con[i], mul_var_enc[i],c='green',marker='o')
    elif mul_var_con[i] < mul_var_enc[i]:
        plt.scatter(mul_var_con[i], mul_var_enc[i],c='red',marker='x')
        print(dataset[i])        
    else:
        plt.scatter(mul_var_con[i], mul_var_enc[i],c='blue',marker='s')

plt.xlabel("multiple variation concatenated")
plt.ylabel("multiple variation encoder")
# plt.plot([0,1],[0,1],'--',transform=plt.gca().transAxes)
# plt.legend(['Tie','Win','Loss'])
plt.plot([0,1],[0,1],'--')
plt.show()




print()




for i in range(len(sin_var_con)):
    if sin_var_con[i] > mul_var_con[i]:
        plt.scatter(sin_var_con[i], mul_var_con[i],c='green',marker='o')
    elif sin_var_con[i] < mul_var_con[i]:
        plt.scatter(sin_var_con[i], mul_var_con[i],c='red',marker='x')
        # print(dataset[i])
        # print("loss ",dataset[i])        

    else:
        plt.scatter(sin_var_con[i], mul_var_con[i],c='blue',marker='s')
        # print("equal ",dataset[i])

plt.xlabel("single variation concatenated")
plt.ylabel("multiple variation concatenated")
# plt.plot([0,1],[0,1],'--',transform=plt.gca().transAxes)
# plt.legend(['Tie','Win','Loss'])
plt.plot([0,1],[0,1],'--')
plt.show()

print()

for i in range(len(sin_var_con)):
    if sin_var_enc[i] > mul_var_enc[i]:
        plt.scatter(sin_var_enc[i], mul_var_enc[i],c='green',marker='o')
    elif sin_var_enc[i] < mul_var_enc[i]:
        plt.scatter(sin_var_enc[i], mul_var_enc[i],c='red',marker='x')
        # print("loss ",dataset[i])        
    else:
        plt.scatter(sin_var_enc[i], mul_var_enc[i],c='blue',marker='s')
        # print("equal ",dataset[i])

plt.xlabel("single variation encoder")
plt.ylabel("multiple variation encoder")
# plt.plot([0,1],[0,1],'--',transform=plt.gca().transAxes)
# plt.legend(['Tie','Win','Loss'])
plt.plot([0,1],[0,1],'--')
plt.show()

print()


# ssssssssssssssss


# print(ucr_accuracy[dataset[i]])

for i in range(len(sin_var_con)):
    if sin_var_con[i] >1- ucr_accuracy[dataset[i]]:
        plt.scatter(sin_var_con[i], 1- ucr_accuracy[dataset[i]],c='green',marker='o')
        print("loss ",dataset[i])
    elif sin_var_con[i] < 1- ucr_accuracy[dataset[i]]:
        plt.scatter(sin_var_con[i], 1- ucr_accuracy[dataset[i]],c='red',marker='x')
    else:
        plt.scatter(sin_var_con[i], 1- ucr_accuracy[dataset[i]],c='blue',marker='s')

plt.xlabel("single variation concatenated")
plt.ylabel("UCR DTW")
# plt.plot([0,1],[0,1],'--',transform=plt.gca().transAxes)
plt.plot([0,1],[0,1],'--')
# plt.legend(['Tie','Win','Loss'])
plt.show()

print()


for i in range(len(sin_var_con)):
    if mul_var_con[i] > 1- ucr_accuracy[dataset[i]]:
        plt.scatter(mul_var_con[i], 1- ucr_accuracy[dataset[i]],c='green',marker='o')
        print("loss ",dataset[i])        
    elif mul_var_con[i] < 1- ucr_accuracy[dataset[i]]:
        plt.scatter(mul_var_con[i], 1- ucr_accuracy[dataset[i]],c='red',marker='x')
    else:
        plt.scatter(mul_var_con[i], 1- ucr_accuracy[dataset[i]],c='blue',marker='s')

plt.xlabel("multiple variation concatenated")
plt.ylabel("UCR DTW")
# plt.plot([0,1],[0,1],'--',transform=plt.gca().transAxes)
# plt.legend(['Tie','Win','Loss'])
plt.plot([0,1],[0,1],'--')
plt.show()




print()

# zzzzzzzzzzzzzzzz


for i in range(len(sin_var_con)):
    if mul_var_enc[i] > 1- ucr_accuracy[dataset[i]]:
        plt.scatter(mul_var_enc[i], 1- ucr_accuracy[dataset[i]],c='green',marker='o')
        print("loss ",dataset[i])        
    elif mul_var_enc[i] < 1- ucr_accuracy[dataset[i]]:
        plt.scatter(mul_var_enc[i], 1- ucr_accuracy[dataset[i]],c='red',marker='x')
        # print(dataset[i])

    else:
        plt.scatter(mul_var_enc[i], 1- ucr_accuracy[dataset[i]],c='blue',marker='s')
        # print("equal ",dataset[i])

plt.xlabel("multiple variation encoder")
plt.ylabel("UCR DTW")
# plt.plot([0,1],[0,1],'--',transform=plt.gca().transAxes)
# plt.legend(['Tie','Win','Loss'])
plt.plot([0,1],[0,1],'--')
plt.show()

print()

for i in range(len(sin_var_con)):
    if sin_var_enc[i] > 1- ucr_accuracy[dataset[i]]:
        plt.scatter(sin_var_enc[i], 1- ucr_accuracy[dataset[i]],c='green',marker='o')
        print("loss ",dataset[i])        
    elif sin_var_enc[i] < 1- ucr_accuracy[dataset[i]]:
        plt.scatter(sin_var_enc[i], 1- ucr_accuracy[dataset[i]],c='red',marker='x')
    else:
        plt.scatter(sin_var_enc[i], 1- ucr_accuracy[dataset[i]],c='blue',marker='s')
        # print("equal ",dataset[i])

plt.xlabel("single variation encoder")
plt.ylabel("UCR DTW")
# plt.plot([0,1],[0,1],'--',transform=plt.gca().transAxes)
# plt.legend(['Tie','Win','Loss'])
plt.plot([0,1],[0,1],'--')
plt.show()





