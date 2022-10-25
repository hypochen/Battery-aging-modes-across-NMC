# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 13:04:59 2021

@author: WALKCM2
"""
#% Main file for analyzing LAM data

import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import copy
import numpy as np
from matplotlib import ticker
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# custom libraries
from openPouchSummary import openPouchSummaryFcn 
from fcnCBCdict import fcnCBCdict               
from detrendCBCdict import detrendCBCdict    
from createDataframeforLAM import createDataframeforLAM
#%% Loading Pouch Summary
# Describes testing rates, pack information, and which cells to use

file_location = r"C:\path\Pouch cell_summary.xlsx"
pouch_summary = openPouchSummaryFcn(file_location)
print("Loaded pouch summary")

#%% Extracting cycle-by-cycle (CBC) data into a dictionary

packs = list(set(pouch_summary['Pack']))                       # Which packs are we working with.
rootdir = r"C:\path"                                           # Where are the files located?
pack_dict = fcnCBCdict(rootdir,pouch_summary)                  # Cycles through folders extracting CBC for each pack and cell.

pack_save = copy.deepcopy(pack_dict) # copy original pack_dict, so you don't have to reload when testing.
print("Extracted CBC data")

#%% Detrending CBC data
# Removing RPT's seen in original data by treating them as a seasonal effect.

pack_dict = detrendCBCdict(pack_dict,period=50)
print("Detrended CBC data")
# Plotting the difference between the original data with RPT's and the trended data
plt.figure()
plt.plot(pack_dict['P462']['04']['Cycle'],pack_dict['P462']['04']['EOC_Cell'],label='extrap')
plt.plot(pack_save['P462']['04']['Cycle'],pack_save['P462']['04']['EOC_Cell'],label='original')
plt.ylabel('EOC_Cell');plt.xlabel('Cycle'); plt.title('P462 Cell 04');plt.ylim([3.85,3.9]);plt.legend()

#%% Preprocessing PackDict into a DataFrame of useful features
df = createDataframeforLAM(pack_dict,pouch_summary)
print("Data converted into a DataFrame")

#%% Dropping rows that do not contain %LAM information
df = df[df['%LAM '] != 0]
df['estimate'] = [[] for r in range(len(df))] # adding a column to store estimates

columns=['C_rate','max_cycle','capacity_AR', 'capacity_MA', 'capacity_max', 'capacity_min','capacity_slope','capacity_x2slope','capacity_x1slope','capacity_min2last','capacity_max2last','CE_AR', 'CE_MA', 'CE_max', 'CE_min','CE_slope','CE_x2slope','CE_x1slope','CE_min2last','CE_max2last','EOC_AR', 'EOC_MA', 'EOC_max','EOC_slope','EOC_x2slope','EOC_x1slope', 'EOC_min','EOC_min2last','EOC_max2last','EOD_AR', 'EOD_MA', 'EOD_max', 'EOD_min','EOD_slope','EOD_x2slope','EOD_x1slope','EOD_min2last','EOD_max2last']

df_importance = pd.DataFrame(data=[],columns=columns)

#%% Repeat training and testing with shuffled data for distribution of errors

cycles = 100

RF_acc = np.zeros(cycles)
SVR_acc = np.zeros(cycles)
LinReg_acc = np.zeros(cycles)

for ii in range(0,cycles):
    #% Break up into training and testing data w/ cross validation
    
    train, test = train_test_split(df, test_size=0.1,random_state=ii)
    #X = train.loc[:,columns]
    #X = train.loc[:,['CE_AR', 'CE_MA','EOC_slope','EOD_min','EOD_x1slope']]
    X = train.loc[:,['capacity_min2last','capacity_slope','EOD_min']]
    Y = train['%LAM ']
    
    #X_test = test.loc[:,columns]
    #X_test = test.loc[:,['CE_AR', 'CE_MA','EOC_slope','EOD_min','EOD_x1slope']]
    X_test = test.loc[:,['capacity_min2last','capacity_slope','EOD_min']]
    Y_test = test['%LAM ']
    
    #% Random Forest
    RF_model = RandomForestRegressor(max_depth=None, random_state=0, n_estimators=100)
    RF_model.fit(X, Y)
    
    RF_guess = RF_model.predict(X_test)
    RF_accuracy = mean_squared_error(Y_test, RF_guess, squared=False)
    RF_acc[ii] = RF_accuracy
    
    for jj in range(0,len(Y_test)):
        df['estimate'][Y_test.index[jj]].append(RF_guess[jj])  # recording the guesses
        
    #df_imp = pd.DataFrame(data=[RF_model.feature_importances_],columns=columns)
    #df_importance = pd.concat([df_importance, df_imp])
    
# =============================================================================
#     
#     # Finding the best hyperparameters using a grid search.
#     scaler = StandardScaler()
#     Xs = scaler.fit_transform(X)
#     C_range = np.logspace(-2, 10, 13)
#     epsilon_range = np.logspace(-9, 3, 13)
#     param_grid = dict(epsilon=epsilon_range, C=C_range)
#     cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
#     grid = GridSearchCV(SVR(), param_grid=param_grid, cv=cv)
#     grid.fit(Xs, Y)
#     #print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
#     
#     SVR_model = make_pipeline(StandardScaler(), SVR(kernel='rbf',C=grid.best_params_['C'],epsilon =grid.best_params_['epsilon'], random_state=0))
# =============================================================================
    
    X_svr = train.loc[:,['CE_AR', 'CE_MA','EOC_slope','EOD_min','EOD_x1slope']]
    X_svr_test = test.loc[:,['CE_AR', 'CE_MA','EOC_slope','EOD_min','EOD_x1slope']]
    
    SVR_model = SVR(kernel='rbf',C=100,epsilon =0.01)
    SVR_model.fit(X_svr, Y)
    
    SVR_guess = SVR_model.predict(X_svr_test)
    SVR_accuracy = mean_squared_error(Y_test, SVR_guess, squared=False) 
    SVR_acc[ii] = SVR_accuracy
    
    # Linear Regression
    reg = LinearRegression().fit(X, Y)
    LinReg_guess = reg.predict(X_test)
    LinReg_accuracy = mean_squared_error(Y_test, LinReg_guess, squared=False)
    LinReg_acc[ii] = LinReg_accuracy

#% Print Results from multi-run cycle
print('RF Accuracy of '+str(np.mean(RF_acc))[0:6]+ ' +- '+str(np.std(RF_acc))[0:6])
print('SVR Accuracy of '+str(np.mean(SVR_acc))[0:6]+ ' +- '+str(np.std(SVR_acc))[0:6])
print('LinReg Accuracy of '+str(np.mean(LinReg_acc))[0:6]+ ' +- '+str(np.std(LinReg_acc))[0:6])

#%% Adding Colors to DataFrame

hex_color_dict = {
    'green':'#008000',
    'goldenrod2':'#EEB422',
    'blue':'#0000FF',
    'red1':'#FF0000',
    'aliceblue':'#F0F8FF',
    'cornsilk1':'#FFF8DC',
    'lavenderblush1':'#FFF0F5'
    }

color_dict = {
    'P462':hex_color_dict['blue'],
    'P492':hex_color_dict['red1'],
    'P531':hex_color_dict['goldenrod2'],
    'P540':hex_color_dict['green'],
    'P533':hex_color_dict['blue']
    }


shape_dict = {
    'P462':'^',
    'P492':'s',
    'P531':'o',
    'P540':'v',
    'P533':'^'
    }

df['colors'] = [color_dict[pack] for pack in df['pack']]  # adding colors based on pack and chemical composition
df['shapes'] = [shape_dict[pack] for pack in df['pack']]  # adding colors based on pack and chemical composition

#%% Plotting %LAM actual versus %LAM estimated
means = []
stds = []
for ii in range(0,len(df['estimate'])):
    means.append(np.mean(df['estimate'][df['estimate'].index[ii]]))
    stds.append(np.std(df['estimate'][df['estimate'].index[ii]])*2) # times by 2 to have 95% confidence

plt.figure()
plt.rcParams['font.family'] = ['Arial']
plt.rcParams.update({'font.size': 34})
plt.errorbar(df['%LAM '],means,yerr=stds,ecolor=df['colors'],ls = "None",label='RF estimate w/ 95% confidence')
plt.legend(fontsize=31)
plt.plot(df['%LAM '],df['%LAM '],c='black') # 1-1 line

# Marker can only have 1 style at a time.
m = df['shapes']
unique_markers = ['s','x','P','v']  # or yo can use: np.unique(m)
for um in unique_markers:
    mask = m == um 
    # mask is now an array of booleans that can be used for indexing  
    plt.scatter(np.array(df['%LAM '][mask]), np.array(means)[np.flatnonzero(mask).astype(int)], marker=um)

plt.xlabel('Actual %LAM',fontsize=34); plt.ylabel('Estimated %LAM',fontsize=34);

#%% List of best variables

FI_i = list(np.mean(df_importance))
FI_x=list(X.columns)
zipped = zip(FI_i,FI_x)
FI_zip = sorted(zipped, key = lambda x: x[0])

#%% Plotting Variables Versus %LAM

var_to_compare = 'capacity_min2last'

theta = np.polyfit(df['%LAM '], df[var_to_compare], 1)
y_line = theta[1] + theta[0] * df['%LAM '] # detemining best fit line
correlation_matrix = np. corrcoef(df['%LAM '], df[var_to_compare]);correlation_xy = correlation_matrix[0,1];
r_squared = correlation_xy**2.
plt.figure();plt.scatter(df['%LAM '],df[var_to_compare])
plt.plot(df['%LAM '],y_line, label='$\mathregular{R^2}$ = '+str(r_squared)[0:4]); plt.legend()
plt.ylabel('capacity tilt') ; plt.xlabel('%LAM')


var_to_compare = 'capacity_slope'

theta = np.polyfit(df['%LAM '], df[var_to_compare], 1)
y_line = theta[1] + theta[0] * df['%LAM '] # detemining best fit line
correlation_matrix = np. corrcoef(df['%LAM '], df[var_to_compare]);correlation_xy = correlation_matrix[0,1];
r_squared = correlation_xy**2.
plt.figure();plt.scatter(df['%LAM '],df[var_to_compare])
plt.plot(df['%LAM '],y_line, label='$\mathregular{R^2}$ = '+str(r_squared)[0:4]); plt.legend()
plt.ylabel('capacity slope') ; plt.xlabel('%LAM')


var_to_compare = 'EOD_min'

theta = np.polyfit(df['%LAM '], df[var_to_compare], 1)
y_line = theta[1] + theta[0] * df['%LAM '] # detemining best fit line
correlation_matrix = np. corrcoef(df['%LAM '], df[var_to_compare]);correlation_xy = correlation_matrix[0,1];
r_squared = correlation_xy**2.
plt.figure();plt.scatter(df['%LAM '],df[var_to_compare])
plt.plot(df['%LAM '],y_line, label='$\mathregular{R^2}$ = '+str(r_squared)[0:4]); plt.legend()
plt.ylabel('EODV min') ; plt.xlabel('%LAM')

#%%
var_to_compare = 'EOC_min2last'

theta = np.polyfit(df['%LAM '], df[var_to_compare], 1)
y_line = theta[1] + theta[0] * df['%LAM '] # detemining best fit line
correlation_matrix = np. corrcoef(df['%LAM '], df[var_to_compare]);correlation_xy = correlation_matrix[0,1];
r_squared = correlation_xy**2.
plt.figure(figsize=(3, 3),dpi=300);plt.scatter(df['%LAM '],df[var_to_compare])
plt.plot(df['%LAM '],y_line, label='$\mathregular{R^2}$ = '+str(r_squared)[0:4]); plt.legend()
plt.ylabel('EOCV tilt') ; plt.xlabel('%LAM')


var_to_compare = 'EOD_min2last'

theta = np.polyfit(df['%LAM '], df[var_to_compare], 1)
y_line = theta[1] + theta[0] * df['%LAM '] # detemining best fit line
correlation_matrix = np. corrcoef(df['%LAM '], df[var_to_compare]);correlation_xy = correlation_matrix[0,1];
r_squared = correlation_xy**2.
plt.figure();plt.scatter(df['%LAM '],df[var_to_compare])
plt.plot(df['%LAM '],y_line, label='$\mathregular{R^2}$ = '+str(r_squared)[0:4]); plt.legend()
plt.ylabel('EODV tilt') ; plt.xlabel('%LAM')


#%% Plotting figures with subplots

var_to_compare = 'capacity_min2last'
theta = np.polyfit(df['%LAM '], df[var_to_compare], 1)
y_line0 = theta[1] + theta[0] * df['%LAM '] # detemining best fit line
correlation_matrix = np. corrcoef(df['%LAM '], df[var_to_compare]);correlation_xy = correlation_matrix[0,1];
r_squared0 = correlation_xy**2.

var_to_compare = 'EOC_min2last'
theta = np.polyfit(df['%LAM '], df[var_to_compare], 1)
y_line1 = theta[1] + theta[0] * df['%LAM '] # detemining best fit line
correlation_matrix = np. corrcoef(df['%LAM '], df[var_to_compare]);correlation_xy = correlation_matrix[0,1];
r_squared1 = correlation_xy**2.

var_to_compare = 'EOD_min2last'
theta = np.polyfit(df['%LAM '], df[var_to_compare], 1)
y_line2 = theta[1] + theta[0] * df['%LAM '] # detemining best fit line
correlation_matrix = np. corrcoef(df['%LAM '], df[var_to_compare]);correlation_xy = correlation_matrix[0,1];
r_squared2 = correlation_xy**2.

markersize = 400
# Plotting all the subplots
fig, axes = plt.subplots(2, 2,figsize=(20,15))
axes[0, 0].yaxis.set_major_locator(ticker.MaxNLocator(4))
axes[0, 1].yaxis.set_major_locator(ticker.MaxNLocator(4))
axes[1, 0].yaxis.set_major_locator(ticker.MaxNLocator(4))
axes[1, 1].yaxis.set_major_locator(ticker.MaxNLocator(4))

#axes[0, 0].scatter(df['%LAM '],df['capacity_min2last'],color=df['colors'], s=markersize,marker=df['shapes'])
zero = axes[0, 0].scatter(df['%LAM '][df['shapes'] == 's'],df['capacity_min2last'][df['shapes'] == 's'], c=color_dict['P492'], s=markersize, marker=shape_dict['P492'])
one = axes[0, 0].scatter(df['%LAM '][df['shapes'] == 'o'], df['capacity_min2last'][df['shapes'] == 'o'],  c=color_dict['P531'], s=markersize, marker=shape_dict['P531'])
two = axes[0, 0].scatter(df['%LAM '][df['shapes'] == '^'], df['capacity_min2last'][df['shapes'] == '^'],  c=color_dict['P462'], s=markersize, marker=shape_dict['P462'])
three = axes[0, 0].scatter(df['%LAM '][df['shapes'] == 'v'], df['capacity_min2last'][df['shapes'] == 'v'],  c=color_dict['P540'], s=markersize, marker=shape_dict['P540'])
axes[0, 0].plot(df['%LAM '], y_line0, label='$\mathregular{R^2}$ = '+str(r_squared0)[0:4]); axes[0, 0].legend()
plt.setp(axes[0,0], xlabel='%LAM') ; plt.setp(axes[0,0], ylabel='Capacity fade (mA-h)')


#axes[0, 1].scatter(df['%LAM '],df['EOC_min2last'],color=df['colors'], s=markersize,marker=df['shapes'])
zero = axes[0, 1].scatter(df['%LAM '][df['shapes'] == 's'],df['EOC_min2last'][df['shapes'] == 's'], c=color_dict['P492'], s=markersize, marker=shape_dict['P492'])
one = axes[0, 1].scatter(df['%LAM '][df['shapes'] == 'o'], df['EOC_min2last'][df['shapes'] == 'o'],  c=color_dict['P531'], s=markersize, marker=shape_dict['P531'])
two = axes[0, 1].scatter(df['%LAM '][df['shapes'] == '^'], df['EOC_min2last'][df['shapes'] == '^'],  c=color_dict['P462'], s=markersize, marker=shape_dict['P462'])
three = axes[0, 1].scatter(df['%LAM '][df['shapes'] == 'v'], df['EOC_min2last'][df['shapes'] == 'v'],  c=color_dict['P540'], s=markersize, marker=shape_dict['P540'])
axes[0, 1].plot(df['%LAM '], y_line1, label='$\mathregular{R^2}$ = '+str(r_squared1)[0:4]); axes[0, 1].legend()
plt.setp(axes[0,1], xlabel='%LAM') ; plt.setp(axes[0,1], ylabel='EOCV tilt (V)')

#axes[1, 0].scatter(df['%LAM '],df['EOD_min2last'],color=df['colors'], s=markersize,marker=df['shapes'])
zero = axes[1, 0].scatter(df['%LAM '][df['shapes'] == 's'],df['EOD_min2last'][df['shapes'] == 's'], c=color_dict['P492'], s=markersize, marker=shape_dict['P492'])
one = axes[1, 0].scatter(df['%LAM '][df['shapes'] == 'o'], df['EOD_min2last'][df['shapes'] == 'o'],  c=color_dict['P531'], s=markersize, marker=shape_dict['P531'])
two = axes[1, 0].scatter(df['%LAM '][df['shapes'] == '^'], df['EOD_min2last'][df['shapes'] == '^'],  c=color_dict['P462'], s=markersize, marker=shape_dict['P462'])
three = axes[1, 0].scatter(df['%LAM '][df['shapes'] == 'v'], df['EOD_min2last'][df['shapes'] == 'v'],  c=color_dict['P540'], s=markersize, marker=shape_dict['P540'])
axes[1, 0].plot(df['%LAM '], y_line2, label='$\mathregular{R^2}$ = '+str(r_squared2)[0:4]); axes[1, 0].legend()
plt.setp(axes[1,0], xlabel='%LAM') ; plt.setp(axes[1,0], ylabel='EODV tilt (V)')

# =============================================================================
# shape_dict = {
#     0:'s',
#     1:'x',
#     2:'P'
#     }
# 
# color_dict = {
#     0:'navy',
#     1:'firebrick',
#     2:'gold'
#     }
# =============================================================================


zero = axes[1,1].scatter(df['%LAM '][df['shapes'] == 's'],np.array(means)[df['shapes'] == 's'], c=color_dict['P492'], s=markersize, marker=shape_dict['P492'])
one = axes[1,1].scatter(df['%LAM '][df['shapes'] == 'o'], np.array(means)[df['shapes'] == 'o'],  c=color_dict['P531'], s=markersize, marker=shape_dict['P531'])
two = axes[1,1].scatter(df['%LAM '][df['shapes'] == '^'], np.array(means)[df['shapes'] == '^'],  c=color_dict['P462'], s=markersize, marker=shape_dict['P462'])
three = axes[1,1].scatter(df['%LAM '][df['shapes'] == 'v'], np.array(means)[df['shapes'] == 'v'],  c=color_dict['P540'], s=markersize, marker=shape_dict['P540'])
plt.xlabel('Actual %LAM',fontsize=34); plt.ylabel('Estimated %LAM',fontsize=34);
axes[1, 1].errorbar(df['%LAM '],means,yerr=stds,ecolor=df['colors'],ls = "None",label='RF estimate w/ 95% confidence')
axes[1, 1].plot(df['%LAM '],df['%LAM '], color='black')
plt.tight_layout()


#%%
theta = np.polyfit(df['%LAM '], means, 1)
y_line1 = theta[1] + theta[0] * df['%LAM '] # determining best fit line
correlation_matrix = np. corrcoef(df['%LAM '], means);correlation_xy = correlation_matrix[0,1];
r_squared_est = correlation_xy**2














