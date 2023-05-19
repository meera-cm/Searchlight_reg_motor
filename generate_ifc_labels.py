#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 10:59:13 2020

@author: Meera
"""

import os
import json


subjects = [1,2,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,29,30,31,32,33,34,35]
file_paths = []

# path = os.path.join(os.getcwd(), "data", "SPL_mask_extracted")
os.chdir("data")
os.chdir("Motor_mask_extracted")
# 

for root, dirs, files in os.walk(os.getcwd()):
    for f in files:
        # if f == "ifc_data.json" or f == "labels.json":
        if f == "motor_data.json" or f == "labels.json":

            continue
        file_path = os.path.join(root, f)
        file_paths.append(file_path)
        

# Remove extra .DS_store that was an invisible item included in the list.
# file_paths.pop(0)
print(len(file_paths))
# file_paths.sort()
print(file_paths[0],file_paths[-1])

if len(subjects) != len(file_paths):
    print("error!")
else:
    print("correct length!")
    subject_dict = dict(zip(subjects, file_paths)) 
    for k, v in subject_dict.items():
        print(k, v)
    # write json
    with open("motor_data.json", "w") as fp:
        json.dump(subject_dict, fp)
    # print(subject_dict.keys())