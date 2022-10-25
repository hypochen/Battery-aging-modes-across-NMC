# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 10:13:02 2021

@author: WALKCM2
"""

# Getting cycle-by-cycle data for each cell in each pack
# Returns a dictionary with such.

import os
import pandas as pd 

def fcnCBCdict(rootdir,pouch_summary):
    pack_dict = {}  # making a dictionary to store cell data in.
    packs = list(set(pouch_summary['Pack']))
    for subdir, dirs, files in os.walk(rootdir):
        if subdir[len(rootdir)+1:len(rootdir)+5] in packs:                         # finding folders related to the packs
            pack = subdir[len(rootdir)+1:len(rootdir)+5]                           # working with one pack at a time
            cells = pouch_summary["Cell number"][pouch_summary["Pack"] == pack]    # getting cells related to each pack nuumber
            pack_dict[pack] = {}                                                   # making a dictionary for each pack
            
            for cell in cells:
                cell = str(int(cell)).zfill(2)                                     # making a cell number be a string with length of 2 by front padding.
                pack_dict[pack][cell] = {}                                         # the cells within the packs.
                
                for file in files:
                    measurement_type = file[:-6]
                    cell_on_filename = file[-6:-4]
                    
                    if cell_on_filename == cell:
                        data = pd.read_csv(os.path.join(subdir, file))             # get data
                        data = data.iloc[:,1:]                                     # drop useless first column
                        data = data.rename(columns={data.columns[0]:"Cycle", data.columns[1]:measurement_type}) # rename columns to represent measurement type
                        pack_dict[pack][cell].update(data)                         # updates the dictionary with new information from the most recent data set (not cycles)
    return pack_dict