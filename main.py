from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import numpy as np

from NN_searchlight import Net
from dataset import SearchlightDataset
from utils import *


def main():
    
    # Hyperparameters
    
    batch_size = 32
    num_output_features = 1
    epochs = 800
    # learning_rate = 2 * 1e-3
    learning_rate = 1 * 1e-2

    #0.9
    momentum = 0.9
    log_tensorboard_freq = 5
    log_model_freq = 1000
    weight_decay = 0.2
    delete_logs = True
   
    # Delete all files on logs folder
    if delete_logs:
        log_path = os.path.join(os.getcwd(), "logs")
        delete_files(log_path)
  
    json_ifc_labels = os.path.join(os.getcwd(), "data", "labels.json")
    with open(json_ifc_labels, "r") as f:
        # key: subject ID, value: tms_effect in seconds
        labels_dict = json.load(f)        
    subjects = list(labels_dict.keys())
    
    # First for loop
    for t, test_subj_ID in enumerate(subjects):
        # if t > 15:
        #     exit()
        
       
        print("testing subject: ", test_subj_ID)
        train = SearchlightDataset(is_test_set=False,
                                    test_subj_ID=test_subj_ID)
        
        test = SearchlightDataset(is_test_set=True,
                                   test_subj_ID=test_subj_ID)

        train_loader = DataLoader(dataset=train,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=1)
        
        test_loader = DataLoader(dataset=test,
                                 batch_size=1,
                                 shuffle=True,
                                 num_workers=1)
    
        writer = SummaryWriter("logs")
        # data_iter = iter(train_loader)
        # input_img, labels = data_iter.next()    
            
        net = Net()
        # print(net)
        
        # writer.add_graph(net, input_img) 
    
        # Loss - mean squared error (MSE); Optimiser - stochastic gradient descent (SGD)
        criterion = nn.MSELoss()
        # optimiser = optim.SGD(net.parameters(), 
        #                       lr=learning_rate, 
        #                       momentum=momentum)
        # optimiser = optim.Adam(net.parameters(),
        #                         lr=learning_rate, 
                                # weight_decay=weight_decay)
        optimiser = optim.RMSprop(net.parameters(),
                                  lr=learning_rate)
    
        train_log_dict = {"epochs": [], "loss": []}
        test_log_dict = {"epochs": [], "loss": []}
        
        # writer.close()
        # first for loop to divide data into training set and test set
    
        # second for loop iterates through epochs
        for e in range(epochs):
            
            # final for loop iterates through batches
            for batch_input, batch_target in train_loader:  
                optimiser.zero_grad()
                batch_out = net.forward(batch_input)
                loss = criterion(batch_out, batch_target)
                loss.backward()
                optimiser.step()
                
                # print("batch out : ", batch_out)
                # print("batch target : ", batch_target)
                
                # log training loss for epoch on tensorboard
                print("training loss: ", loss.item())
                
                # append eopchs and loss
                train_log_dict["epochs"].append(e)
                train_log_dict["loss"].append(loss.item())
               
                
            # test every 5 epochs
            if e % log_tensorboard_freq == 0:
                for batch_input, batch_target in test_loader:
                    optimiser.zero_grad()
                    batch_out = net.forward(batch_input)
                    loss = criterion(batch_out, batch_target)
                    
                    writer.add_scalars("losses %s" % str(test_subj_ID), {
                       "test loss %s" % str(test_subj_ID): loss,
                       "train loss %s" % str(test_subj_ID): train_log_dict["loss"][e]}, e)
                    print("test loss: ", loss.item())
                    # print(batch_out, batch_target)
                
                    test_log_dict["epochs"].append(e)
                    test_log_dict["loss"].append(loss.item())

            # # save model
            # if e % log_model_freq == 0:
            #     save_net_path =  os.path.join(os.getcwd(), "models/model_%d.pth" % e)
            #     torch.save(net.state_dict(), save_net_path)

        print("Training completed successfully")
        train_log_df = pd.DataFrame.from_dict(train_log_dict)
        train_log_df.to_csv("logs/train_log_%s.csv" % str(test_subj_ID))
        test_log_df =  pd.DataFrame.from_dict(test_log_dict)
        test_log_df.to_csv("logs/test_log_%s.csv" % str(test_subj_ID))
        print("Logs saved to logs folder")
        # exit()

if __name__ == "__main__":
    main()