#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 10:45:43 2023

@author: mariohabibfathi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read the data from the CSV file
data = pd.read_csv('outputs/FCNEncoderDecoder/run4/jigsaw/resultes/elsa7/dubdub.csv')

# Extracting the dataset names, number of segments, and number of permutations from the first column
data['Dataset Name'] = data['Test Type'].str.split('  ').str[0]
data['Number of Segments'] = data['Test Type'].str.extract(r'(\d+) segments')[0].astype(int)
data['Number of Permutations'] = data['Test Type'].str.extract(r'(\d+) permutations')[0].astype(int)


# filtered_data = data[data['Test Type'].str.contains('nobottleneck')]
# filtered_data = data[data['Test Type'].str.contains('Beef')]


# Create a pivot table to reshape the data for the 3D plot
pivot_data = data.pivot(index='Number of Segments', columns='Number of Permutations', values='Accuracy')
segments = np.array(pivot_data.index)
permutations = np.array(pivot_data.columns)
accuracy_values = pivot_data.values

# Plot the 3D surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(segments, permutations)
ax.plot_surface(X, Y, accuracy_values, cmap='viridis')

# Set axis labels and title
ax.set_xlabel('Number of Segments')
ax.set_ylabel('Number of Permutations')
ax.set_zlabel('Accuracy')
ax.set_title('Accuracy vs Number of Segments and Permutations')

plt.show()
