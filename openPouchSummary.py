# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 13:06:13 2021

@author: WALKCM2
"""
import pandas as pd 

# Opening excel file and sorting its contents.
# Returning a dataframe with pack and cell metadata that should be in the LAM analysis.

def openPouchSummaryFcn(file_location):
    # Open xlsx
    file_data = pd.read_excel(file_location, sheet_name = None, header = 1)
    
    # Sort through each excel page
    for key in file_data:
        
        # Keep only data associated with "YES" for suggested to use as training
        file_data[key]['Suggest to use as training'] = file_data[key]['Suggest to use as training'].fillna(value = "NO") # replacing nan's with NO's to be removed
        file_data[key].drop(file_data[key][file_data[key]['Suggest to use as training']=="NO"].index, inplace= True)     # drop rows with a NO
        
        # Adding Pack number to dataframe
        pack_list = [key] * len(file_data[key])
        file_data[key]['Pack'] = pack_list
    
    df = pd.concat(file_data) # Make a single nice, neat dataframe.
    df = df.fillna(value = 0) # replacing nan's with 0's
    df = df.reset_index()     # reindexing so it starts back at 0
    return df

