# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 13:04:59 2021

@author: WALKCM2
"""
#% Main file for analyzing LLI/LAM classification

import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import copy
from statsmodels.tsa.arima.model import ARIMA
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.pyplot import figure
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.decomposition import PCA

# custom libraries
from openPouchSummary import openPouchSummaryFcn 
from fcnCBCdict import fcnCBCdict               
from detrendCBCdict import detrendCBCdict    
from createDataframeFromPackDictV2 import createDataframeFromPackDictV2  
#%% Loading Pouch Summary
# Describes testing rates, pack information, and which cells to use

# rainbow excel file location
file_location = r"C:\path\Pouch cell_summary.xlsx"
pouch_summary = openPouchSummaryFcn(file_location)
print("Loaded pouch summary")

#%% Extracting cycle-by-cycle (CBC) data into a dictionary

# CBC data location for rootdir
packs = list(set(pouch_summary['Pack']))                       # Which packs are we working with.
rootdir = r"C:\path"                                           # Where are the files located?
pack_dict = fcnCBCdict(rootdir,pouch_summary)                  # Cycles through folders extracting CBC for each pack and cell.
pack_save = copy.deepcopy(pack_dict) # copy original pack_dict, so you don't have to reload when testing.
print("Extracted CBC data")
        
#%% Detrending CBC data

# Removing RPT's seen in original data by treating them as a seasonal effect.
pack_dict = detrendCBCdict(pack_dict,period=50)
print("Detrended CBC data")

#%% Remove temperature dips at end of pack 1 cells.

pack = 'P492' # This pack contained extreme outliers
for cell in pack_dict[pack]:
    for measurement in pack_dict[pack][cell]:
        y = pack_dict[pack][cell][measurement]
        y[326:394] = np.NAN
        y[509:562] = np.NAN
        pack_dict[pack][cell][measurement]=y.interpolate() 

#%% Plotting the difference between the original data with RPT's and the trended data

plt.figure()
cell = '04'
plt.plot(pack_save['P462'][cell]['Cycle'],pack_save['P462'][cell]['EOC_Cell'],label='original')
plt.plot(pack_dict['P462'][cell]['Cycle'],pack_dict['P462'][cell]['EOC_Cell'],label='processed')
plt.ylabel('EOCV');plt.xlabel('Cycle'); plt.legend();# plt.title('P492 Cell 24');
plt.ylim([3.85,3.9]);

#%% Preprocessing PackDict into a DataFrame of useful features

df = createDataframeFromPackDictV2(pack_dict,pouch_summary,data_length=25)
print("Data converted into a DataFrame")
Acc_summary_df = pd.DataFrame(data={'Cycles':[],'RF_acc':[],'RF_std':[],'SVM_acc':[],'SVM_std':[],'KNN_acc':[],'KNN_std':[],'RF_LDA_acc':[],'RF_LDA_std':[]})

for jj in range(1,25):
    data_length = jj*25 
        
# =============================================================================
# for jj in range(0,1):
#     data_length = 600
# =============================================================================
    
    df_new = createDataframeFromPackDictV2(pack_dict,pouch_summary,data_length=data_length) # making df for new data length
    df_new = df_new.drop(['LLI','pack','cell'], axis=1) # dropping LLI so the answer isn't in the training data
    df_new=df_new.rename(columns=lambda s: s + str(data_length)) # renaming columns based on variable & data length
    df = pd.concat([df,df_new],ignore_index=False, axis=1)
    print("Data converted into a DataFrame")
    
    #% Repeat training and testing with shuffled data for distribution of errors  
    cycles = 40
    
    RF_acc = np.zeros(cycles)
    ET_acc = np.zeros(cycles)
    SVM_acc = np.zeros(cycles)
    RF_LDA_acc = np.zeros(cycles)
    KNN_acc = np.zeros(cycles)
    mis_class_df = pd.DataFrame(data={'pack':[],'cell':[]})
    
    for ii in range(0,cycles):
        #% Break up into training and testing data w/ cross validation
        
        train, test = train_test_split(df, test_size=0.2,random_state=ii)
        
        #X = train.loc[:,['CE_AR', 'CE_MA','EOC_slope','EOD_min','EOD_x1slope']]
        X = train.loc[:,train.columns !='LLI']
        X = X.loc[:,X.columns !='pack']
        X = X.loc[:,X.columns !='cell']
        Y = train['LLI']
        
        #X_test = test.loc[:,['CE_AR', 'CE_MA','EOC_slope','EOD_min','EOD_x1slope']]
        X_test = test.loc[:,train.columns !='LLI']
        X_test = X_test.loc[:,X_test.columns !='pack']
        X_test = X_test.loc[:,X_test.columns !='cell']
        Y_test = test['LLI']
        
        #% Random Forest
        RF_model = RandomForestClassifier(max_depth=None,criterion = "gini", random_state=0, n_estimators=100)
        RF_model.fit(X, Y)
        RF_guess = RF_model.predict(X_test)
        RF_accuracy = RF_model.score(X_test,Y_test)
        #print('RF Accuracy of '+str(RF_accuracy)[0:6]+ ' from '+str(len(Y_test))+' examples')
        RF_acc[ii] = RF_accuracy
        
        #% Finding misclassified samples
        mis_labeled = test.loc[Y_test.tolist() != RF_guess,['pack','cell']]
        mis_class_df = pd.concat([mis_class_df,mis_labeled])
        
        
        # RF + LDA
# =============================================================================
#         pca = PCA(n_components=5,svd_solver='full')
#         X_lda = pca.fit_transform(X)
#         X_test_lda = pca.transform(X_test)
# =============================================================================
        
        lda = LinearDiscriminantAnalysis(n_components=2)
        X_lda = lda.fit(X, Y).transform(X)
        X_test_lda = lda.transform(X_test)
        lda = QuadraticDiscriminantAnalysis()
        lda.fit(X, Y)
        lda.predict(X_test)
        
        RF_LDA_model = RandomForestClassifier(max_depth=None,criterion = "gini", random_state=0, n_estimators=100)
        RF_LDA_model.fit(X_lda, Y)
        RF_LDA_guess = RF_LDA_model.predict(X_test_lda)
        RF_LDA_accuracy = RF_LDA_model.score(X_test_lda,Y_test)
        RF_LDA_acc[ii] = RF_LDA_accuracy
      
        #% SVM for classification
        # Finding the best hyperparameters using a grid search.
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        C_range = np.logspace(-2, 10, 13)
        gamma_range = np.logspace(-9, 3, 13)
        param_grid = dict(gamma=gamma_range, C=C_range)
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=41)
        grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
        grid.fit(Xs, Y)
        #print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
        SVM_model = make_pipeline(StandardScaler(), SVC(kernel='rbf',C=grid.best_params_['C'],gamma =grid.best_params_['gamma'], random_state=0))
        SVM_model.fit(X, Y)
        SVM_guess = SVM_model.predict(X_test)
        SVM_accuracy = SVM_model.score(X_test,Y_test)
        #print('SVM Accuracy of '+str(svm_accuracy)[0:6]+ ' from '+str(len(Y_test))+' examples')
        SVM_acc[ii] = SVM_accuracy
        
        #% K-Nearest Neighbors
        KNN_model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=10))
        KNN_model.fit(X, Y)
        
        KNN_guess = KNN_model.predict(X_test)
        KNN_accuracy = KNN_model.score(X_test,Y_test)
        #print('KNN Accuracy of '+str(KNN_accuracy)[0:6]+ ' from '+str(len(Y_test))+' examples')
    
        KNN_acc[ii] = KNN_accuracy
    

    #% Print Results from multi-run cycle
    print('Data length is '+str(data_length))
    print('RF Accuracy of '+str(np.mean(RF_acc))[0:6]+ ' +- '+str(np.std(RF_acc))[0:6])
    print('RF_LDA Accuracy of '+str(np.mean(RF_LDA_acc))[0:6]+ ' +- '+str(np.std(RF_LDA_acc))[0:6])
    print('SVM Accuracy of '+str(np.mean(SVM_acc))[0:6]+ ' +- '+str(np.std(SVM_acc))[0:6])
    print('KNN Accuracy of '+str(np.mean(KNN_acc))[0:6]+ ' +- '+str(np.std(KNN_acc))[0:6])
    Acc_summary_df =pd.concat([Acc_summary_df,pd.DataFrame(data={'Cycles':[data_length],'RF_acc':[np.mean(RF_acc)],'RF_std':[np.std(RF_acc)],'SVM_acc':[np.mean(SVM_acc)],'SVM_std':[np.std(SVM_acc)],'KNN_acc':[np.mean(KNN_acc)],'KNN_std':[np.std(KNN_acc)],'RF_LDA_acc':[np.mean(RF_LDA_acc)],'RF_LDA_std':[np.std(RF_LDA_acc)]})],ignore_index = True)

#%% List of best variables

FI_i = list(RF_model.feature_importances_)
FI_x=list(X.columns)
zipped = zip(FI_i,FI_x)
FI_zip = sorted(zipped, key = lambda x: x[0])

#%% Misclassified cells

g = mis_class_df.groupby(['pack', 'cell'])
total_misclasses = g.agg({'cell' :'count'})
total_misclasses = total_misclasses.rename(columns={"cell": "# of mislabels"})

#%% Exploratory

X = df.drop(['LLI'], axis = 1)
Y = df['LLI']
X = pd.get_dummies(X, prefix_sep='_')
Y = LabelEncoder().fit_transform(Y)
X = StandardScaler().fit_transform(X)


def forest_test(X, Y):
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, 
                                                        test_size = 0.10, 
                                                        random_state = 0)
    start = time.process_time()
    trainedforest = RandomForestClassifier(n_estimators=700).fit(X_Train,Y_Train)
    print(time.process_time() - start)
    predictionforest = trainedforest.predict(X_Test)
    print(confusion_matrix(Y_Test,predictionforest))
    print(classification_report(Y_Test,predictionforest))

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
PCA_df = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2'])
PCA_df = pd.concat([PCA_df, df['LLI']], axis = 1)
PCA_df['LLI'] = LabelEncoder().fit_transform(PCA_df['LLI'])
PCA_df.head()

figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')

classes = [2, 1, 0]
colors = ['k','r', 'b']
markr = ["o","^","s"]
for clas, color in zip(classes, colors):
    plt.scatter(PCA_df.loc[PCA_df['LLI'] == clas, 'PC1'], 
                PCA_df.loc[PCA_df['LLI'] == clas, 'PC2'], 
                c = color, marker=markr[clas])
    
plt.xlabel('Principal Component 1', fontsize = 12)
plt.ylabel('Principal Component 2', fontsize = 12)
plt.title('2D PCA', fontsize = 15)
plt.legend(['Li plating','SEI + less LAM', 'SEI + more LAM'])
plt.grid()

#%%
X_Reduced, X_Test_Reduced, Y_Reduced, Y_Test_Reduced = train_test_split(X_pca, Y, 
                                                                        test_size = 0.20, 
                                                                        random_state = 101)
trainedforest = RandomForestClassifier(n_estimators=700).fit(X_Reduced,Y_Reduced)

x_min, x_max = X_Reduced[:, 0].min() - 1, X_Reduced[:, 0].max() + 1
y_min, y_max = X_Reduced[:, 1].min() - 1, X_Reduced[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = trainedforest.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
plt.contourf(xx, yy, Z,cmap=plt.cm.coolwarm, alpha=0.4)
plt.scatter(X_Reduced[:, 0], X_Reduced[:, 1], c=Y_Reduced, s=20, edgecolor='k')
plt.xlabel('Principal Component 1', fontsize = 12)
plt.ylabel('Principal Component 2', fontsize = 12)
plt.title('Random Forest', fontsize = 15)
plt.show()

#%%
pca = PCA(n_components=20,svd_solver='full')
X_pca = pca.fit_transform(X)
print(pca.explained_variance_)
forest_test(X_pca, Y)

#%%
lda = LinearDiscriminantAnalysis(n_components=2)

# run an LDA and use it to transform the features
X_lda = lda.fit(X, Y).transform(X)
print('Original number of features:', X.shape[1])
print('Reduced number of features:', X_lda.shape[1])

y=Y
target_names = ['Li plating','SEI + less LAM', 'SEI + more LAM']

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
print(
    "explained variance ratio (first two components): %s"
    % str(pca.explained_variance_ratio_)
)

plt.figure()
colors = ["navy", "turquoise", "darkorange"]
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(
        X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("PCA of Battery dataset")

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(
        X_r2[y == i, 0], X_r2[y == i, 1], alpha=0.8, color=color, label=target_name
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("LDA of Battery dataset")

plt.show()


#%% Plotting LDA + Random Forest
X_Reduced, X_Test_Reduced, Y_Reduced, Y_Test_Reduced = train_test_split(X_lda, Y, 
                                                                        test_size = 0.20, 
                                                                        random_state = 101)
trainedforest = RandomForestClassifier(n_estimators=700).fit(X_Reduced,Y_Reduced)
x_min, x_max = X_Reduced[:, 0].min() - 1, X_Reduced[:, 0].max() + 1
y_min, y_max = X_Reduced[:, 1].min() - 1, X_Reduced[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = trainedforest.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
plt.contourf(xx, yy, Z,colors=['#F0F8FF', '#FFF8DC', '#FFF0F5'], alpha=0.4)
#cmap=plt.cm.hsv
colors=['#808080', '#A0A0A0', '#C0C0C0']
target_names = ['SEI + more LAM','SEI + less LAM','Li plating']
target_labelled = []
for ii in range(0,len(Y_Reduced)):
    target_labelled.append(target_names[Y_Reduced[ii]])
scatter = plt.scatter(X_Reduced[:, 0], X_Reduced[:, 1], c=Y_Reduced, s=20, edgecolor='k')
plt.xlabel('LDA 1', fontsize = 12)
plt.ylabel('LDA 2', fontsize = 12)
plt.title('LDA + Random Forest', fontsize = 15)
plt.legend(handles=scatter.legend_elements()[0], labels=target_names)
#legend1 = plt.legend(*scatter.legend_elements(),loc="lower left", title="Classes")
plt.show()

#%% Plotting Classification Accuracy with subplots

Y_RF = RF_acc
Y_SVM = SVM_acc
Y_KNN = KNN_acc

shape_dict = {
    0:'s',
    1:'o',
    2:'^'
    }

color_dict = {
    0:'#FF0000',
    1:'#EEB422',
    2:'#0000FF'
    }

# Figure 1
markersize = 300
edgewidth = 2
# Plotting all the subplots
fig, axes = plt.subplots(1, 2,figsize=(20,10))


# Figure
X_Reduced, X_Test_Reduced, Y_Reduced, Y_Test_Reduced = train_test_split(X_lda, Y,test_size = 0.20,random_state = 101)
trainedforest = RandomForestClassifier(n_estimators=700).fit(X_Reduced,Y_Reduced)
x_min, x_max = X_Reduced[:, 0].min() - 1, X_Reduced[:, 0].max() + 1
y_min, y_max = X_Reduced[:, 1].min() - 5, X_Reduced[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = trainedforest.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
axes[0].contourf(xx, yy, Z,cmap=plt.cm.gist_rainbow, alpha=.05, extend='both')
target_names = ['SEI + more LAM','SEI + less LAM','Li plating']
target_labelled = []
for ii in range(0,len(Y_Reduced)):
    target_labelled.append(target_names[Y_Reduced[ii]])
#axes[1].scatter(X_Reduced[ii, 0], X_Reduced[ii, 1], label=[Y_Reduced[ii]], c=color_dict[Y_Reduced[ii]], s=30, edgecolor='k',marker=shape_dict[Y_Reduced[ii]])
plt.setp(axes[0], xlabel='LDA 1') ; plt.setp(axes[0], ylabel='LDA 2')

zero = axes[0].scatter(X_Reduced[Y_Reduced == 0, 0], X_Reduced[Y_Reduced == 0, 1], label=target_names[0], c=color_dict[0], s=markersize, linewidths = edgewidth, edgecolor='k',marker=shape_dict[0])
one = axes[0].scatter(X_Reduced[Y_Reduced == 1, 0], X_Reduced[Y_Reduced == 1, 1], label=target_names[1], c=color_dict[1], s=markersize,linewidths = edgewidth, edgecolor='k',marker=shape_dict[1])
two = axes[0].scatter(X_Reduced[Y_Reduced == 2, 0], X_Reduced[Y_Reduced == 2, 1], label=target_names[2], c=color_dict[2], s=markersize,linewidths = edgewidth, edgecolor='k',marker=shape_dict[2])

#handles=scatter.legend_elements()[0],labels=target_names,
axes[0].legend(loc="lower right")
#axes[1].set_ylim([-3,4]);

# Next Figure 
X_cycles = range(25,625,25)

axes[1].plot(X_cycles,Y_RF,'o',label ='RF', linestyle='solid',linewidth=3);
axes[1].plot(X_cycles,Y_SVM,'o',label ='SVM', linestyle='dotted',linewidth=3);
axes[1].plot(X_cycles,Y_KNN,'o',label ='KNN', linestyle='dashed',linewidth=3); 
axes[1].legend()
plt.setp(axes[1], xlabel='Cycles') ; plt.setp(axes[1], ylabel='Accuracy (%)')
plt.tight_layout()

#%% Plotting examples of feature extraction

plt.figure(figsize=(15,15))
series = pack_dict['P462']['05']['EOD_Cell']
plt.plot(series, label='Data')

model = ARIMA(series, order=(1,0,1))
model_fit = model.fit()
predictions = list()
for t in range(2,len(series)):
    model = ARIMA(series[0:t], order=(1,0,1))
    model_fit = model.fit()
    obs = model_fit.forecast()
    predictions.append(obs[t])


#%% Plotting types of estimates for EODV

plt.figure()
plt.rcParams['font.family'] = ['Arial']
plt.rcParams.update({'font.size': 34})
plt.legend(fontsize=31)
plt.figure(figsize=(10,10))
series = pack_dict['P462']['05']['EOD_Cell']

plt.scatter(range(0,len(series)),series, label='Experimental data')
plt.plot(model_fit.forecasts[0][1:],'r', label='ARMA forecasting 1-step ahead', linestyle='solid',linewidth=3)

p = np.polyfit(range(0,len(series)),series,2) # 2nd degree polynomial, whole dataset
plt.plot(range(0,len(series)),np.polyval(p, range(0,len(series))),color='green',label='2nd degree polynomial')

p = np.polyfit(range(350,len(series)),series[350:],1) # 1st degree polynomial, last 100
plt.plot(range(350,len(series)),np.polyval(p, range(350,len(series))),color='k',linestyle='dashed',linewidth=3,label='Slope of last 100 points')

plt.scatter(series.idxmin(axis=0),series.min(axis=0),marker='>',label='Minimum',linewidth=10)
plt.scatter(series.idxmax(axis=0),series.max(axis=0),color='brown',marker='+',label='Maximum',linewidth=10)
#plt.vlines(x=series.idxmin(axis=0), ymin=series.min(axis=0), ymax=series[449],label='EODV tilt',linewidth=5)
plt.vlines(x=449, ymin=series.min(axis=0), ymax=series[449],label='EODV tilt',linewidth=4, ls=':')

plt.xlabel('Cycle');plt.ylabel('EODV (V)')
plt.legend()














