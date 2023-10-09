#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:54:37 2023

@author: mariohabibfathi
"""

import os
import subprocess
# import uno
import sys
import re

# sys.path.append('/usr/lib/python3/dist-packages/')

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]










num = 20
# set the path to the directory containing the PDF files
pdf_dir = "images/DTW1/"
pdf_dir = 'images/multipledata/run2/random_samples/'
pdf_dir = 'images/multipledata/run2/random_samples_concat/'
pdf_dir = 'images/multipledata/run2/random_samples_separated/'
pdf_dir = f'images/FCNEncoderDecoder/run3/{num}_Segments/bottlenecktest/random_samples_concat/'

# get a list of the PDF files in the directory
pdf_files = os.listdir(pdf_dir)

# sort the files alphabetically
pdf_files.sort(key=natural_keys)

# create a list to hold the groups of files
file_groups = []

# group the files into sets of two
for i in range(0, len(pdf_files), 12):
    file_groups.append(pdf_files[i:i+12])

# set the output filename
output_file = "merged.odt"

# create a list to hold the temporary PDF files
temp_files = []

# iterate over the file groups and merge them into temporary PDF files
for group in file_groups:
    # set the temporary filename
    temp_file = f"{group[0][:-4]}-temp.pdf"
    temp_files.append(temp_file)

    # build the command to merge the PDF files using Ghostscript
    # cmd = f"gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -sOutputFile=merge2/2Parts_only" + " ".join([os.path.join(pdf_dir, f) for f in group])
    cmd = f"gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -sOutputFile=merge2/{num}_segment_bottleneck_test{temp_file} " + " ".join([os.path.join(pdf_dir, f) for f in group])

    # run the command
    subprocess.run(cmd, shell=True)
try:
    
    # build the command to merge the temporary PDF files into a single ODT file
    cmd = f"unoconv --python /usr/bin/python3 --unopath /usr/lib/libreoffice/program --exec /usr/lib/libreoffice/program/soffice.bin -f odt -e PageRange=1-10 --output={output_file} " + " ".join(temp_files)
    
    # run the command
    subprocess.run(cmd, shell=True)
    
    # delete the temporary PDF files
        # for temp_file in temp_files:
        #     os.remove(temp_file)
except subprocess.CalledProcessError as e:
    print("Error :",e)

# print a message to indicate completion
print(f"All PDF files merged into {output_file}")