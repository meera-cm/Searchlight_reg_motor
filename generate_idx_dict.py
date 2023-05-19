#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 12:08:32 2020

@author: Meera
"""

import numpy as np
import pandas as pd
import json
import os


def main():
    # xyz_df_path = os.path.join(os.getcwd(), "data", "xyz_mask_coords.xlsx")
    xyz_df_path = os.path.join(os.getcwd(), "data", "Motor_coords", "xyz_Motor_coords.xlsx")
    
    # Save as pd df
    xyz_df = pd.read_excel(xyz_df_path, header=None)
    
    idx_dict = {}
    
    # distance_array = np.zeros((614, 614))
    # s_array = np.zeros((614, 614))

    # MT
    distance_array = np.zeros((739, 739))
    s_array = np.zeros((739, 739))


    for col_idx in range(len(xyz_df.columns)):
        searchlight_ls, distance_ls, s_list = extract_searchlight(col_idx, xyz_df)
        # idx_dict.update({col_idx:searchlight_ls})
        
        idx_dict[col_idx] = searchlight_ls
        
        distance_ls = np.array(distance_ls)
        distance_array[col_idx, :] = distance_ls
        
        s_list = np.array(s_list)
        s_array[col_idx, :] = s_list
        
        
    # s_df = pd.DataFrame(s_array)

    # filepath = 's_python.xlsx'

    # s_df.to_excel(filepath, index=False)
        

    #     ## convert your array into a dataframe
    # distance_df = pd.DataFrame(distance_array)

    # filepath = 'distances_python.xlsx'

    # distance_df.to_excel(filepath, index=False)
        
    # Save dict in json
    # with open("idx_dict.json", "w") as fp:
        
    # MT    
    with open("Motor_idx_dict.json", "w") as fp:

        json.dump(idx_dict, fp)
    # print(idx_dict.items())

    
def extract_searchlight(centre_col_idx, xyz_df, radius=8, num_voxels=64):
    # Centre point of searchlight dtype=list
    point_centre = list(xyz_df.iloc[:, centre_col_idx])
    # print("point centre is: ", point_centre)    
    # searchlight_list = []

    searchlight_list_idx = []
    distance_list = []
    ls = []
    s_list = []
    for col_idx in range(len(xyz_df.columns)):
        point_current = list(xyz_df.iloc[:, col_idx])
        # print("point current is: ", point_current)
        distance = compute_distance(point_centre, point_current)
        distance_list.append(distance)

        if distance < radius:
            s_list.append(1)
            
            searchlight_list_idx.append(col_idx)
            
        else:
            s_list.append(0)
            
            
    return searchlight_list_idx, distance_list, s_list


def compute_distance(point1, point2):
    """
    Computes the distance between two points given their xyz coordinates
    
    input params: list1 and list2
    input type: two lists
    
    returns distance between the two points

    """
    point1_arr = np.array(point1)
    point2_arr = np.array(point2)
    distance = np.linalg.norm(point1_arr - point2_arr)
    # print("distance: ", distance)

    
    return distance


if __name__ == "__main__":
    main()