#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 10:33:00 2023

@author: mariohabibfathi
"""

import pandas as pd

df = pd.read_csv('outputs/FCNEncoderDecoder/run4/jigsaw/resultes/Send/Autoencoder_Resultes_Detailed.csv')

datasets = ['SmoothSubspace','UMD','ECG200','Beef']



# part = df[0:267]


for data in datasets:
    
    extracted_rows = df[df['Test Type'].str.startswith(data)]
    extracted_rows['avg'] = extracted_rows['RMSE'] + extracted_rows['DTW avg values']
    min_row_both = extracted_rows.loc[extracted_rows['avg'].idxmin()]
    min_row_RMSE = extracted_rows.loc[extracted_rows['RMSE'].idxmin()]
    min_row_DTW = extracted_rows.loc[extracted_rows['DTW avg values'].idxmin()]

    print(f"minimum both {min_row_both['Test Type']} and a value of {extracted_rows['avg'].min()}")
    print(f"minimum RMSE {min_row_RMSE['Test Type']} and a value of {extracted_rows['RMSE'].min()}")
    print(f"minimum DTW {min_row_DTW['Test Type']} and a value of {extracted_rows['DTW avg values'].min()}")
    print()
    max_row_both = extracted_rows.loc[extracted_rows['avg'].idxmax()]
    max_row_RMSE = extracted_rows.loc[extracted_rows['RMSE'].idxmax()]
    max_row_DTW = extracted_rows.loc[extracted_rows['DTW avg values'].idxmax()]
    print(f"max both {max_row_both['Test Type']} and a value of {extracted_rows['avg'].max()}")
    print(f"max RMSE {max_row_RMSE['Test Type']} and a value of {extracted_rows['RMSE'].max()}")
    print(f"max DTW {max_row_DTW['Test Type']} and a value of {extracted_rows['DTW avg values'].max()}")
    print()
    