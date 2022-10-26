# Battery-aging-modes-across-NMC

###  Introduction
Last updated by Bor-Rong (Hypo) Chen and Cody M. Walker.  

Raw battery data and codes for battery aging mode classification.    

The raw dataset consists of 44 NMC/Graphite single layer pouch cells. The data provided include cycle-by-cycle capacity, Coulombic efficiency, end of charge voltage (EOCV), and end of discharge voltage (EODV). 

###  Summary of the cells
A summary of the 44 cells' information, including design parameters, cycling conditions, major aging modes, and experimentally obtained %LAM_PE, can be found in `Pouch cell_summary.xlsx`.  

### Battery data overview 
Stored in the folder `Battery raw data.zip`.  

Cycle-by-cycle battery data, including capacity, Coulombic efficiency, end of charge voltage (EOCV), and end of discharge voltage (EODV), are stored in folders named by the pack number and design:
* `P462_NMC532_R2 design`
* `P492_NMC532_R1 design`
* `P531_NMC811_R1 design`
* `P533_NMC532_R2 design`
* `P540_NMC811_R2 design`

(R1 = L_low and R2 = L_moderate design for electrodes)  

The cycle-by-cycle data are in the format of .csv:
* capacity: `Capacity_CellXX.csv`
* Coulombic efficiency: `CE_CellXX.csv`
* End of charge voltage: (EOCV)`EOC_CellXX.csv`
* End of discharge voltage: (EODV)`EOD_CellXX.csv`

(XX indicates cell number)  

###  Code overview
Download `Battery raw data.zip` and `Pouch cell_summary.xlsx` into a directory of your choice.   

All of the codes used in data processing and analysis can be found in  `code` folder.  

Call `Main_LLI_LAM_Classification.py` and `Main_LAM_estimation.py` to process and analyze the battery data, including data grabbing and pre-processing, creation of a dataframe, data analysis, and plotting. Please change the file directory to fit your local file structure.  
* `Main_LLI_LAM_Classification.py` will classify the cells into Li plating, SEI formation + less LAM_PE, and SEI formation + more LAM_PE.  
* `Main_LAM_estimation.py` will perform a regression to estimate %LAM_PE.  

The following is a library of codes that will be run by `Main_LLI_LAM_Classification.py` and `Main_LAM_estimation.py`.  

* `openPouchSummary.py` selects the cells to serve as training data sets.

* `fcnCBCdict.py` grabs cycle-by-cycle data for each cell in each pack. 

* `detrendCBCdict.py` removes spikes caused by RPTs in the raw battery data. This is done by treating them as seasonal effects and removing them with Seasonal Decomposition of Time Series with period.    

* `createDataframeFromPackDictV2.py` finds trends within series to be used as predictor variables for Decision Tree Classification.   

* `createDataframeforLAM.py`is a replica of `createDataframeFromPackDictV2.py`, but includes the regression analysis of %LAM_PE.  
