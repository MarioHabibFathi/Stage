#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 10:43:10 2023

@author: mariohabibfathi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read data from experiment1.csv (with bottleneck layer)
data_experiment1 = pd.read_csv('experiment1.csv')

# Read data from experiment2.csv (without bottleneck layer)
data_experiment2 = pd.read_csv('experiment2.csv')

# Function to process the data and extract relevant information
def process_data(data):
    data['Dataset Name'] = data['Dataset'].str.split('  ').str[0]
    data['Number of Segments'] = data['Dataset'].str.extract(r'(\d+) segments')[0].astype(int)
    data['Number of Permutations'] = data['Dataset'].str.extract(r'(\d+) permutation')[0].astype(int)
    return data.pivot(index='Number of Segments', columns='Number of Permutations', values='Accuracy')

# Process the data for both experiments
pivot_data_experiment1 = process_data(data_experiment1)
pivot_data_experiment2 = process_data(data_experiment2)

# Get the segments, permutations, and accuracy values for both experiments
segments = np.array(pivot_data_experiment1.index)
permutations = np.array(pivot_data_experiment1.columns)
accuracy_values_experiment1 = pivot_data_experiment1.values
accuracy_values_experiment2 = pivot_data_experiment2.values

# Plot the 3D surface plot for Experiment 1
fig = plt.figure(figsize=(10, 5))

ax1 = fig.add_subplot(121, projection='3d')
X1, Y1 = np.meshgrid(segments, permutations)
ax1.plot_surface(X1, Y1, accuracy_values_experiment1, cmap='viridis')

ax1.set_xlabel('Number of Segments')
ax1.set_ylabel('Number of Permutations')
ax1.set_zlabel('Accuracy')
ax1.set_title('Experiment 1: Accuracy vs Number of Segments and Permutations')

# Plot the 3D surface plot for Experiment 2
ax2 = fig.add_subplot(122, projection='3d')
X2, Y2 = np.meshgrid(segments, permutations)
ax2.plot_surface(X2, Y2, accuracy_values_experiment2, cmap='plasma')

ax2.set_xlabel('Number of Segments')
ax2.set_ylabel('Number of Permutations')
ax2.set_zlabel('Accuracy')
ax2.set_title('Experiment 2: Accuracy vs Number of Segments and Permutations')

plt.tight_layout()
plt.show()
