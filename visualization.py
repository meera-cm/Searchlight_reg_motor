#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 16:10:30 2020

@author: Meera
"""

import pandas as pd
import seaborn as sns


path_test = "/Users/Meera/Downloads/run-losses 1_test loss 1-tag-losses 1.csv"
path_train = "/Users/Meera/Downloads/run-losses 1_train loss 1-tag-losses 1.csv"

test_df = pd.read_csv(path_test)
train_df = pd.read_csv(path_train)
# plot with seaborn

# sns.lmplot(x='Step', y='Value', kind="line", data=df)
# g = sns.relplot(x="Value", y="Step", kind="line", data=df)
# g.fig.autofmt_xdate()
ax = sns.relplot(x='Step',y='Value', kind="line", data=train_df)
ax.fig.autofmt_xdate()

