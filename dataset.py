import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
import pandas as pd
from pandas import DataFrame
from utils import *
import random


class SearchlightDataset(Dataset):


    def __init__(self, is_test_set=False, test_subj_ID=0):
        """

        Returns
        -------
        None.

        """
        
        # subjects = [1,2,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,
        #             24,25,27,28,29,30,31,32,33,34,35]
        self.is_test_set = is_test_set
        # Import xyz excel 
        # xyz_df_path = os.path.join(os.getcwd(), "data", "xyz_mask_coords.xlsx")
        
        # MT
        # xyz_df_path = os.path.join(os.getcwd(), "data", "MT", "xyz_MT_coords.xlsx")

        # Motor mask
        xyz_df_path = os.path.join(os.getcwd(), "data", "Motor_coords", "xyz_Motor_coords.xlsx")
        self.xyz_df = pd.read_excel(xyz_df_path, header=None)
        
        # Import imval file
        # prepimval_path = os.path.join(os.getcwd(), "data", "prepimval.csv")
        # self.prepimval_df = pd.read_csv(prepimval_path, header=None)

        # MT
        prepimval_path = os.path.join(os.getcwd(), "data", "Motor_coords", "Motor_prepimval.xlsx")
        
        
        self.prepimval_df = pd.read_excel(prepimval_path, header=None)
        
        # Labels
        json_ifc_labels = os.path.join(os.getcwd(), "data", "labels.json")
        with open(json_ifc_labels, "r") as f:
            # key: subject ID, value: tms_effect in seconds
            self.labels_dict = json.load(f)
        # Load idx_dict json
        # json_idx_dict = os.path.join(os.getcwd(), "data", "idx_dict.json")
        
        # MT idx_dict
        json_idx_dict = os.path.join(os.getcwd(), "data", "Motor_coords", "Motor_idx_dict.json")
        
        with open (json_idx_dict, "r") as f:
            self.centre_searchlight_dict = json.load(f)
        
        # leave one out
        test_subj_ID = str(test_subj_ID)
        
        subjects = list(self.labels_dict.keys())
        indices = list(range(len(subjects)))
        subj_idx_dict = dict(zip(subjects, indices))
        test_idx = subj_idx_dict[test_subj_ID]
        
        if is_test_set:
            print("generating test set")
            self.labels_dict = {k:v for (k,v) in self.labels_dict.items() if k == test_subj_ID}
            subjects = [test_subj_ID]
            
            # prepimval df containing 1 row
            self.prepimval_df = self.prepimval_df.iloc[test_idx]
            
        else:
            print("generating training set")
            subjects = [s for s in subjects if s != test_subj_ID]
            
            del self.labels_dict[test_subj_ID]
            
            # prepimval df containing 29 rows
            self.prepimval_df = self.prepimval_df.drop([self.prepimval_df.index[test_idx]])
            
        indices = list(range(len(subjects)))  
        self.idx_subj_dict = dict(zip(indices, subjects))
        self.len = len(self.idx_subj_dict)
            
        
    def __getitem__(self, index):
        """
        Given an index mapped with subject IDs, returns input and associated 
        label for that index

        Parameters
        ----------
        index : int
            DESCRIPTION: an index mapped with subject ids

        Returns
        -------
        input_img : ndarray
            DESCRIPTION: array 3x32x32 (3 image slices along x,y,z )
        label : float
            DESCRIPTION: total TMS effect on phase duration (time in sec)

        """
        subj_id = self.idx_subj_dict[index]
        xyz_prepimval_df = self.xyz_df.copy()
        
        if self.is_test_set:
            subject_imval = self.prepimval_df
        else:
            subject_imval = self.prepimval_df.iloc[index]
 
        # Append row to xyz        
        xyz_prepimval_df = xyz_prepimval_df.append(subject_imval, ignore_index=True)
        centre_idx = str(np.random.randint(len(xyz_prepimval_df.columns)))
        # centre_idx = str(300)
        searchlight_idx_list = self.centre_searchlight_dict[centre_idx]
        searchlight_array = extract_searchlight(searchlight_idx_list, xyz_prepimval_df)
        
        # array to tensor
        searchlight_tensor = torch.FloatTensor(searchlight_array)
        searchlight_tensor = torch.squeeze(searchlight_tensor)

        # retrieve labels
        label = [self.labels_dict[subj_id]]
        label = torch.FloatTensor(label)
        return searchlight_tensor, label
    
    
    def __len__(self):
        """
        Method returns length of dataset
        
        :return self.len: length of dataset
        :type self.len: int
        """
        return self.len
                            
        
        
        
    
