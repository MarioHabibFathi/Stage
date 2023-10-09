#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 14:29:31 2023

@author: mariohabibfathi
"""

import pandas as pd
import  re
# Load the CSV file into a dataframe
df = pd.read_csv('outputs/FCNEncoderDecoder/run4/jigsaw/resultes/The last resultes/Comaparison_test_models_for_all_permutation_2_to_40_nobottle.csv', skiprows=1)

# # Extract the numeric part from the column
# df['Numeric_Part'] = df['Test Type'].str.extract('(\d+)').astype(int)

# # Sort the dataframe by 'Column_Name' alphabetically and 'Numeric_Part' numerically
# df.sort_values(['Test Type', 'Numeric_Part'], inplace=True)


# # Extract the numeric part from the column
# df['Numeric_Part'] = df['Test Type'].str.extract('(\d+)')
# df['Numeric_Part'] = df['Numeric_Part'].fillna('0').astype(int)

# # Sort the dataframe by 'Test Type' alphabetically and 'Numeric_Part' numerically
# df.sort_values(['Test Type', 'Numeric_Part'], key=lambda x: x.str.lower(), inplace=True)

# # Extract the numeric part from the column
# df['Numeric_Part'] = df['Test Type'].apply(lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0)

# # Sort the dataframe by 'Test Type' alphabetically and 'Numeric_Part' numerically
# df.sort_values(['Test Type', 'Numeric_Part'], key=lambda x: x.str.lower(), inplace=True)


# # Convert 'Test Type' column to string
# df['Test Type'] = df['Test Type'].astype(str)

# # Extract the numeric part from the column
# df['Numeric_Part'] = df['Test Type'].str.extract('(\d+)')
# df['Numeric_Part'] = df['Numeric_Part'].fillna('0').astype(int)

# # Sort the dataframe by 'Test Type' alphabetically and 'Numeric_Part' numerically
# df.sort_values(['Test Type', 'Numeric_Part'], key=lambda x: x.str.lower(), inplace=True)






# # Convert 'Test Type' column to string
# df['Test Type'] = df['Test Type'].astype(str)

# # Extract the numeric part before " segments" in the column
# df['Numeric_Part'] = df['Test Type'].str.extract(r'(\d+) segments', expand=False).astype(int)

# # Sort the dataframe by 'Test Type' alphabetically and 'Numeric_Part' numerically
# df.sort_values(['Test Type', 'Numeric_Part'], key=lambda x: x.str.lower(), inplace=True)









# # Extract the numeric part before " segments" in the column
# df['Numeric_Part'] = df['Test Type'].str.extract(r'(\d+) segments', expand=False)
# df['Numeric_Part'] = pd.to_numeric(df['Numeric_Part'], errors='coerce')
# df['Numeric_Part'] = df['Numeric_Part'].fillna(0).astype(int)

# # Create a temporary column to hold the combined sorting criteria
# df['Sort_Criteria'] = df['Test Type'] + ' ' + df['Numeric_Part'].astype(str)

# # Sort the dataframe by the temporary 'Sort_Criteria' column
# df.sort_values(by='Sort_Criteria', inplace=True)

# # Remove the temporary columns
# df.drop(['Numeric_Part', 'Sort_Criteria'], axis=1, inplace=True)











# Define a custom sorting key function
def sorting_key(value):
    parts = value.split(' ')
    # return (parts[0], int(parts[2]),int(parts[5]))
    return (parts[0], int(parts[2]),int(parts[5]))


# Apply the sorting key function to the 'Test Type' column
df['Sorting_Key'] = df['Test Type'].apply(sorting_key)

# Sort the DataFrame by the 'Sorting_Key' column
df.sort_values('Sorting_Key', inplace=True)

# Reset the index and drop the 'Sorting_Key' column
df.reset_index(drop=True, inplace=True)
df.drop('Sorting_Key', axis=1, inplace=True)




















# # Remove the 'Numeric_Part' column
# df.drop('Numeric_Part', axis=1, inplace=True)

# Save the sorted dataframe to a new CSV file
df.to_csv('outputs/FCNEncoderDecoder/run4/jigsaw/resultes/The last resultes/test.csv', index=False)