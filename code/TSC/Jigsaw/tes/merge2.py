#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 16:28:59 2023

@author: mariohabibfathi
"""

import os
from subprocess import Popen, PIPE

# set the path to the directory containing the PDF files
pdf_dir = "images/DTW4/"

# get a list of the PDF files in the directory
pdf_files = os.listdir(pdf_dir)

# sort the files alphabetically
pdf_files.sort()

# create a list to hold the pairs of files
file_pairs = []

# iterate over the files, pairing them up
for i in range(0, len(pdf_files), 2):
    if i + 1 >= len(pdf_files):
        # if there's an odd number of files, append the last file to the previous pair
        file_pairs[-1].append(pdf_files[i])
    else:
        file_pairs.append([pdf_files[i], pdf_files[i+1]])

# iterate over the file pairs and merge them into a single PDF file
for pair in file_pairs:
    # set the output filename
    output_file = f"{pair[0][:-4]}-{pair[1][:-4]}.pdf"

    # build the command to merge the PDF files using Ghostscript
    cmd = f"gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -sOutputFile=pdfme/{output_file} {pdf_dir}/{pair[0]} {pdf_dir}/{pair[1]}"

    # run the command
    Popen(cmd, shell=True, stdout=PIPE).communicate()
    
    # print a message to indicate progress
    print(f"Merged {pair[0]} and {pair[1]} into {output_file}")