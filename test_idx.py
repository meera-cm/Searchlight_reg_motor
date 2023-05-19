#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 19:36:16 2020

@author: Meera
"""

import json
import os 

json_idx_dict = os.path.join(os.getcwd(), "data", "idx_dict.json")
with open (json_idx_dict, "r") as f:
    centre_searchlight_dict = json.load(f)
        
for k,v in centre_searchlight_dict.items():
    print(len(v))
