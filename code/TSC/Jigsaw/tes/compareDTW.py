#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 17:37:54 2023

@author: mariohabibfathi
"""

from tslearn.utils import to_time_series_dataset
my_first_time_series = [1, 3, 4, 2]
my_second_time_series = [1, 2, 4, 2]
my_third_time_series = [1, 2, 4, 2, 2]
X = to_time_series_dataset([my_first_time_series,
                               my_second_time_series,
                               my_third_time_series])
y = [0, 1, 1]
