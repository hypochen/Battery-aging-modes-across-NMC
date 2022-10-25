# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 13:22:26 2021

@author: WALKCM2
"""

# Creating Dataframe from pack_dict
# Find trends within series to be used as predictor variables for Decision Tree Classification

from statsmodels.tsa.arima.model import ARIMA
import pandas as pd 
import numpy as np

#%%

def createDataframeFromPackDictV2(pack_dict,pouch_summary,data_length):
    df_data = []
    for pack in pack_dict:
        for cell in pack_dict[pack]:
            cell_summary = []
            for measurement in pack_dict[pack][cell]:
                if measurement == "Cycle":
                    cell_summary.append(data_length)
                else:
                    # Fitting ARIMA parameters
                    signal = pack_dict[pack][cell][measurement][0:data_length]
                    model = ARIMA(signal, order=(1,0,1)) # The order can be changed
                    model_fit = model.fit(method_kwargs={"warn_convergence": False})
                    ar = float(model_fit._params_ar) # autoregressive coeff
                    ma = float(model_fit._params_ma) # moving average coeff
                    
                    measurement_max = max(signal) # max
                    measurement_min = min(signal) # min
                    
                    # Slope of last 100 points, fitting 
                    slope1, intercept1 = np.polyfit(signal[-100:].index.values,signal[-100:],1) # Slope of last 100 points
                    x2slope, x1slope, intercept2 = np.polyfit(signal.index.values,signal,2)     # polynomial (x^2) 
                    
                    cell_summary.extend([ar,ma,measurement_max,measurement_min,slope1,x2slope,x1slope])    # The order of the measurements matter [cycle, capacity, CE, EOC, EOD]
            # finding LI plating in pouch_summary
            pack_of_interest = pouch_summary[pouch_summary['level_0'] == pack]
            cell_info = pack_of_interest[pack_of_interest['Cell number'] == int(cell)]
            rate = float(cell_info['C-rate'].values[0][:-1]) # selects the C-rate and removes the C
            
          
            if str(cell_info['Major aging modes'].values) == "['LLI-Li plating']": # checks "LLI-Li plating"
                LLI = 2 # most LLI plating 
            elif str(cell_info['Major aging modes'].values) == "['LLI-SEI formation + less LAM']":  # checks "LLI-SEI"
                LLI = 1 # LLI-SEI formation + less LAM
            elif str(cell_info['Major aging modes'].values) == "['LLI-SEI formation + more LAM']":  # checks "LLI-SEI"
                LLI = 0 # LLI-SEI formation + less LAM
            else:
                LLI = np.nan # LLI-SEI formation + more LAM
                
    
            df_data.append([pack,cell,LLI,rate]+cell_summary)
    df = pd.DataFrame(df_data, columns = ['pack','cell','LLI','C_rate','max_cycle','capacity_AR', 'capacity_MA', 'capacity_max', 'capacity_min','capacity_slope','capacity_x2slope','capacity_x1slope','CE_AR', 'CE_MA', 'CE_max', 'CE_min','CE_slope','CE_x2slope','CE_x1slope','EOC_AR', 'EOC_MA', 'EOC_max','EOC_slope','EOC_x2slope','EOC_x1slope', 'EOC_min','EOD_AR', 'EOD_MA', 'EOD_max', 'EOD_min','EOD_slope','EOD_x2slope','EOD_x1slope'])        
    return df