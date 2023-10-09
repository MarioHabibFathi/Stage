#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 12:11:53 2023

@author: mariohabibfathi
"""

import os
import re
from PyPDF2 import PdfMerger

def sort_and_merge_files(path):
    filenames = os.listdir(path)
    # print(filenames)
    sorted_filenames = sort_filenames(filenames)
    # print(sorted_filenames)
    merger = PdfMerger()
    
    for filename in sorted_filenames:
        file_path = os.path.join(path, filename)
        with open(file_path, 'rb') as file:
            merger.append(file)
    
    output_directory = os.path.dirname(os.path.abspath(path))
    output_file = os.path.join(output_directory, 'merged_file.pdf')
    with open(output_file, 'wb') as output:
        merger.write(output)
    
    print("Merged file created successfully.")

# def sort_filenames(filenames):
#     def key_function(filename):
#         match = re.match(r'(\w+)_(\d+)_(\d+)_segments_(\d+)_permutations_with_(\w+)\.pdf', filename)
#         if match:
#             data = match.group(1)
#             i = int(match.group(2))
#             seg_num = int(match.group(3))
#             permutation_num = int(match.group(4))
#             bot = match.group(5)
#             return (data, i, seg_num, permutation_num, bot)
#         return filename

#     sorted_filenames = sorted(filenames, key=key_function)
#     return sorted_filenames

# def sort_filenames(filenames):
#     def key_function(filename):
#         match = re.match(r'(\w+)_(\d+)_(\d+)_segments_(\d+)_permutations_with_(\w+)\.pdf', filename)
#         if match:
#             data = match.group(1)
#             i = int(match.group(2))
#             seg_num = int(match.group(3))
#             permutation_num = int(match.group(4))
#             bot = match.group(5)
#             return (data, i, seg_num, permutation_num, bot)
#         return filename

#     sorted_filenames = sorted(filenames, key=key_function)
#     return sorted_filenames

# def sort_filenames(filenames):
#     def key_function(filename):
#         match = re.match(r'(\w+)_(\d+)_(\d+)_segments_(\d+)_permutations_with_(\w+)\.pdf', filename)
#         if match:
#             data = match.group(1)
#             i = match.group(2)
#             seg_num = int(match.group(3))
#             permutation_num = int(match.group(4))
#             bot = match.group(5)
#             return (data, i, seg_num, permutation_num, bot)
#         return filename

#     sorted_filenames = sorted(filenames, key=key_function)
#     return sorted_filenames


# def sort_filenames(filenames):
#     def key_function(filename):
#         match = re.match(r'(\w+)_(\d+)_(\d+)_segments_(\d+)_permutations_with_(\w+)\.pdf', filename)
#         if match:
#             data = match.group(1)
#             i = int(match.group(2))
#             seg_num = int(match.group(3))
#             permutation_num = int(match.group(4))
#             bot = match.group(5)
#             return (data, seg_num, permutation_num,  bot ,i)
#         return filename

#     sorted_filenames = sorted(filenames, key=key_function)
#     return sorted_filenames




# def sort_filenames(filenames):
#     def key_function(filename):
#         # match = re.match(r'(\w+)_((?:worst|best)_DTW)_(\d+)_segments_(\d+)_permutations_with_(\w+)\.pdf', filename)
#         # match = re.match(r'(\w+)_(?:best|worst)_DTW_(\d+)_segments_(\d+)_permutations_with_(\w+)\.pdf', filename)
#         # print("Filename:", filename)
#         match = re.match(r'(\w+)_(best|worst)_DTW_(\d+)_segments_(\d+)_permutations_with_(\w+)\.pdf', filename)
#         # print("Matched groups:", match.groups())

#         if match:
#             # print("Matched groups:", match.groups())

#             data = match.group(1)
#             dtw_type = match.group(2)
#             seg_num = int(match.group(3))
#             permutation_num = int(match.group(4))
#             bot = match.group(5)
#             return (data, seg_num, permutation_num, bot, dtw_type)
#         return filename

#     sorted_filenames = sorted(filenames, key=key_function)
#     return sorted_filenames

def sort_filenames(filenames):
    def key_function(filename):
        match = re.match(r'(\w+)_(best|worst)_DTW_class_(\d+)_(\d+)_segments_(\d+)_permutations_with_(\w+)\.pdf', filename)
        if match:
            data = match.group(1)
            dtw_type = match.group(2)
            class_number = int(match.group(3))
            seg_num = int(match.group(4))
            permutation_num = int(match.group(5))
            bot = match.group(6)
            return (data, seg_num, permutation_num, bot, class_number, dtw_type)
        return filename

    sorted_filenames = sorted(filenames, key=key_function)
    return sorted_filenames



path = "images/FCNEncoderDecoder/run4/jigsaw/max_min_by_class/pdf/"
sort_and_merge_files(path)
