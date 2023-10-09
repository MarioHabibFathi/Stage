#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 12:53:02 2023

@author: mariohabibfathi
"""

import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
data = pd.read_csv('outputs/FCNEncoderDecoder/run4/jigsaw/resultes/Autoencoder_Resultes_Simplified.csv')

# # Filter the DataFrame to include only 'nobottleneck' data
# filtered_data = data[data['Test Type'].str.contains('nobottleneck')]

# # Extract the relevant columns
# segments = filtered_data['Test Type'].str.extract(r'(\d+) segments')
# permutations = filtered_data['Test Type'].str.extract(r'(\d+) permutations')
# rmse_values = filtered_data['RMSE']

# # Convert extracted columns to numeric types
# segments = segments.astype(int)
# permutations = permutations.astype(int)

# # Determine the unique segment numbers
# unique_segments = segments[0].unique()
# # unique_segments = segments


# # Iterate over each unique segment number and plot the corresponding permutations
# for segment in unique_segments:
#     # Filter data for the current segment
#     segment_data = filtered_data[segments == segment]
#     segment_permutations = permutations[segments == segment]
#     segment_rmse_values = rmse_values[segments == segment]
    
#     # Create a new plot for the segment
#     plt.figure()
#     plt.title(f'Segment {segment} - RMSE vs Permutations')
#     plt.xlabel('Permutations')
#     plt.ylabel('RMSE')
    
#     # Plot RMSE values for each permutation
#     for permutation, rmse in zip(segment_permutations, segment_rmse_values):
#         plt.plot(permutation, rmse, 'o', label=f'Permutations: {permutation}')
    
#     plt.legend()
#     plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt

# # Read the CSV file
# data = pd.read_csv('outputs/FCNEncoderDecoder/run4/jigsaw/resultes/sortiisa.csv')

# # Filter the data for "nobottleneck" cases
# nobottleneck_data = data[data['Test Type'].str.contains('nobottleneck')]

# # Extract the unique segment numbers
# segment_numbers = nobottleneck_data['Test Type'].str.extract(r'(\d+) segments')[0].unique()

# # Plot the RMSE values for each segment number
# for segment_num in segment_numbers:
#     segment_data = nobottleneck_data[nobottleneck_data['Test Type'].str.contains(f'{segment_num} segments')]
#     permutation_numbers = segment_data['Test Type'].str.extract(r'(\d+) permutations')[0].unique()
    
#     plt.figure()
#     plt.title(f"RMSE values for {segment_num} segments")
#     plt.xlabel("Permutation Number")
#     plt.ylabel("RMSE")
    
#     for permutation_num in permutation_numbers:
#         permutation_data = segment_data[segment_data['Test Type'].str.contains(f'{permutation_num} permutations')]
#         plt.plot(permutation_data.index, permutation_data['RMSE'], label=f"{permutation_num} permutations")
    
#     plt.legend()
#     plt.show()




# import pandas as pd
# import matplotlib.pyplot as plt

# # Read the CSV file into a DataFrame
# data = pd.read_csv('your_file.csv')

# Filter the DataFrame to include only 'nobottleneck' data
# filtered_data = data[data['Test Type'].str.contains('nobottleneck')]

# # Extract the relevant columns
# segments = filtered_data['Test Type'].str.extract(r'(\d+) segments')
# permutations = filtered_data['Test Type'].str.extract(r'(\d+) permutations')
# rmse_values = filtered_data['RMSE']

# # Convert extracted columns to numeric types
# segments = segments.astype(int)
# permutations = permutations.astype(int)

# # Group the filtered data by 'Test Type' and extract the unique segments for each dataset
# grouped_segments = filtered_data.groupby('Test Type')['Test Type'].apply(lambda x: int(x.iloc[0].split()[2]))

# # Iterate over each unique segment and plot the corresponding permutations for each dataset
# for segment in grouped_segments.unique():
#     # Filter data for the current segment
#     segment_data = filtered_data[grouped_segments == segment]
#     segment_permutations = permutations[grouped_segments == segment]
#     segment_rmse_values = rmse_values[grouped_segments == segment]
    
#     # Create a new plot for the segment
#     plt.figure()
#     plt.title(f'Segment {segment} - RMSE vs Permutations')
#     plt.xlabel('Permutations')
#     plt.ylabel('RMSE')
    
#     # Plot RMSE values for each permutation
#     for permutation, rmse in zip(segment_permutations, segment_rmse_values):
#         plt.plot(permutation, rmse, 'o', label=f'Permutations: {permutation}')
    
#     plt.legend()
#     plt.show()

###############

# Filter the DataFrame to include only 'nobottleneck' data
# filtered_data = data[data['Test Type'].str.contains('nobottleneck')]

# # Extract the relevant columns
# segments = filtered_data['Test Type'].str.extract(r'(\d+) segments')
# permutations = filtered_data['Test Type'].str.extract(r'(\d+) permutations')
# rmse_values = filtered_data['RMSE']

# # Convert extracted columns to numeric types
# segments = segments.astype(int)
# permutations = permutations.astype(int)

# # Group the filtered data by 'Test Type' and extract the unique segments for each dataset
# grouped_segments = filtered_data.groupby('Test Type')['Test Type'].apply(lambda x: int(x.iloc[0].split()[2]))

# # Reset index of filtered data
# filtered_data = filtered_data.reset_index(drop=True)

# # Iterate over each unique segment and plot the corresponding permutations for each dataset
# for segment in grouped_segments.unique():
#     # Filter data for the current segment
#     segment_data = filtered_data.loc[grouped_segments[grouped_segments == segment].index]
#     segment_permutations = permutations.loc[grouped_segments[grouped_segments == segment].index]
#     segment_rmse_values = rmse_values.loc[grouped_segments[grouped_segments == segment].index]
    
#     # Create a new plot for the segment
#     plt.figure()
#     plt.title(f'Segment {segment} - RMSE vs Permutations')
#     plt.xlabel('Permutations')
#     plt.ylabel('RMSE')
    
#     # Plot RMSE values for each permutation
#     for permutation, rmse in zip(segment_permutations, segment_rmse_values):
#         plt.plot(permutation, rmse, 'o', label=f'Permutations: {permutation}')
    
#     plt.legend()
#     plt.show()



# Filter the data for "nobottleneck" cases
filtered_data = data[data['Test Type'].str.contains('nobottleneck')]
filtered_data = filtered_data[filtered_data['Test Type'].str.contains('Beef')]

# Extract the segment number and RMSE values    
# segments = filtered_data['Test Type'].str.extract(r'(\d+) segments')

# rmse_values = filtered_data['RMSE']

# # Convert segment numbers to integers
# segments = segments.astype(int)

# # Group the RMSE values by segment number
# grouped_data = rmse_values.groupby(segments)

# # Plot the change in RMSE values with increasing permutations for each segment
# fig, ax = plt.subplots()
# for segment, group in grouped_data:
#     ax.plot(grouped_data.get_group(segment), label=f'Segment {segment}')

# ax.set_xlabel('Permutations')
# ax.set_ylabel('RMSE')
# ax.set_title('Change in RMSE with Increasing Permutations')
# ax.legend()
# plt.show()

seg_nums = filtered_data['Test Type'].str.extract(r'(\d+) segments').astype(int)
seg_num_counts = filtered_data['Test Type'].str.extract(r'(\d+) segments').astype(int).value_counts()
filtered_seg_nums = seg_num_counts[seg_num_counts > 1].index
filtered_data2 = filtered_data[filtered_data['Test Type'].str.extract(r'(\d+) segments').astype(int).isin(filtered_seg_nums)]


# for seg_num in seg_nums.unique():
#     seg_data = filtered_data[filtered_data['Test Type'].str.contains(f'{seg_num} segments')]  # Filter data for the current seg_num
#     plt.plot(seg_data['permutations'], seg_data['RMSE'], label=f"seg_num = {seg_num}")

permutation_nums = filtered_data['Test Type'].str.extract(r'(\d+) permutations').astype(int)

# Plot the change in RMSE values for each specific data
# for index, row in filtered_data.iterrows():
#     seg_num = seg_nums.iloc[index]
#     permutation_num = permutation_nums.iloc[index]
#     rmse = row['RMSE']
    # plt.scatter(permutation_num, rmse, label=f"seg_num = {seg_num}")

