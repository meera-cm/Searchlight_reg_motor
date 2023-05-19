#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 17:00:33 2020

@author: Meera
"""

import numpy as np
import os
import shutil
import random


def extract_searchlight(searchlight_idx_list, xyz_df, 
                        radius=8, num_voxels=64):
    
    append_dummy = True
    
    if len(searchlight_idx_list) >= num_voxels:
        searchlight_idx_list = random.sample(searchlight_idx_list, num_voxels)
        append_dummy = False
    
    # Centre point of searchlight
    searchlight_list = []

    for col_idx in searchlight_idx_list:
        col = list(xyz_df.iloc[:, col_idx])
        searchlight_list.append(col)
   
    if append_dummy:
        dummy_list = [-1, -1, -1, -1]
        num_dummies = num_voxels - len(searchlight_list)
        for i in range(num_dummies):
            searchlight_list.append(dummy_list)

    searchlight_array = np.array(searchlight_list)
    searchlight_array = searchlight_array.reshape(1,-1)
    
    return searchlight_array


def delete_files(path):
    shutil.rmtree(path)
    os.mkdir(path)
    print("all files in deleted from: ", path)