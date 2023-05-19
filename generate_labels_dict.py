#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:26:16 2020

@author: Meera
"""

import os
import numpy as np
import json
import pandas as pd



def normalize_labels(label_list):
    """
    input: list of labels
    
    return: list of normalized values

    """
    normed_list = []
    
    l_max = max(label_list)
    l_min = min(label_list)
    l_range = l_max - l_min
    
    for label in label_list:
        norm_label = (label-l_min)/l_range
        normed_list.append(norm_label)
        
    return normed_list


subjects = [1,2,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,29,30,31,32,33,34,35]

questfile_path = os.path.join(os.getcwd(), "data", "quest_cfs.xlsx")
questfile = pd.read_excel(questfile_path)

label_list = questfile.loc[:,"tms_effect"]

label_list = [l for l in label_list if str(l) != 'nan']
# print(label_list)

label_list = normalize_labels(label_list)


labels_dict = dict(zip(subjects, label_list))
with open("labels.json", "w") as fp:
    json.dump(labels_dict, fp)
# print(labels_dict.items())


        