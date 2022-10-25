# Joule_battery-aging-modes-across-NMC

###  Introduction
Last updated by Bor-Rong (Hypo) Chen and Cody M. Walker.  
Raw battery data and codes for battery aging mode classification.    
The raw dataset consisted of 44 NMC/Gr single layer pouch cells. The data provided include cycle-by-cycle capacity, Columbic efficiency, end of charge voltage (EOCV), and end of discharge voltage (EODV). 

###  Summary of the cells
A summary of the 44 cells' information, including design parameters and cycling conditions, can be found in `Pouch cell_summary.xlsx`.  

### Battery data overview 
Stored in the folder `Battery raw data.zip`.  

Cycle-by-cycle battery data, including capacity, Columbic efficiency, end of charge voltage (EOCV), and end of discharge voltage (EODV), are stored in folders named by the pack number and design:
* `P462_NMC532_R2 design`
* `P492_NMC532_R1 design`
* `P531_NMC811_R1 design`
* `P533_NMC532_R2 design`
* `P540_NMC811_R2 design`

The cycle-by-cycle data are in the format of .csv:
* capacity: `Capacity_CellXX.csv`
* Columbic efficiency: `CE_CellXX.csv`
* End of charge voltage: (EOCV)`EOC_CellXX.csv`
* End of discharge voltage: (EODV)`EOD_CellXX.csv`

###  Code overview
Download `Battery raw data.zip` and `Pouch cell_summary.xlsx` into a directory of your choice.   

All of the codes used in data processing and analysis can be found in  `code` folder.  

Call the `Main_LLI_LAM_Classification.py` and `Main_LAM_estimation.py` to process and analyze the battery data, including data grabbing and pre-processing, creation of a dataframe, data analysis, and plotting. Please change the file directory to fit your local file structure.  
* `Main_LLI_LAM_Classification.py` will classify cells into Li plating, SEI formation, and LAM_PE.  
* `Main_LAM_estimation.py` will do a regression to estimate %LAM_PE.  

The following is a library of codes that will be run by `Main_LLI_LAM_Classification.py` and `Main_LAM_estimation.py`.  

* `openPouchSummary.py` selects the cells to serve as training data sets.

* `fcnCBCdict.py` will grab cycle-by-cycle data for each cell in each pack. 

* `detrendCBCdict.py` will remove spikes caused by RPTs in the raw battery data. This is done by treating them as seasonal effects and removing them with Seasonal Decomposition of Time Series with period.    

* `createDataframeFromPackDictV2.py`will find trends within series to be used as predictor variables for Decision Tree Classification.   

* `createDataframeforLAM.py`is a replica of `createDataframeFromPackDictV2.py`, but include the regression analysis of %LAM.  
