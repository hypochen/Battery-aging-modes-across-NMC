# Joule_battery-aging-modes-across-NMC
Joule_battery aging modes across NMC
###  Introduction
Last updated by  
Bor-Rong (Hypo) Chen and Cody M. Walker
Code and raw data for the aging mode analysis study.    
Dataset consisted of 44 NMC/Gr single layer pouch cells.  
Capacity, Columbic efficiency, end of charge voltage (EOCV), and end of discharge voltage (EODV) are included. 

###  Summary of the cells' information 
A summary of the 44 cells' information can be found in `summary.xlxs`.  
Use `openPouchSummary.py` to select the cells to be analyzed. 

###  Code overview

Set up your file directory 

Call the `Main_LLI_LAM_Classification.py` and `Main_LAM_estimation.py`to process and analyze the data, including data grabbing and preprocessing, then create a dataframe, plus analysis and plotting. 

Use `fcnCBCdict.py` to getting cycle-by-cycle data for each cell in each pack. This is the one code that reads the raw data in the libary. 

Use `createDataframeFromPackDictV2.py`to find trends within series to be used as predictor variables for Decision Tree Classification
Use `createDataframeforLAM.py` this one also includes the analysis of %LAM. The function is all the same with line 17. to find trends within series to be used as predictor variables for Decision Tree Classification
  
Use `detrendCBCdict.py` to remove RPTs by treating them as seasonal effects and removing them with Seasonal Decomposition of Time Series with period    


Use `Main_LLI_LAM_Classification.py`to  
Use `Main_LAM_estimation.py`to  

###  Run the analysis
