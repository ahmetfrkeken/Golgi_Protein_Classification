import pandas as pd
import numpy as np
from lightgbm.sklearn import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

 
 
 

def get_data(data_pd):
    data = data_pd.iloc[:,1:]
    print(data.shape)
     
    label = data_pd.iloc[:,0]
     
    IDs=data_pd.index
    return IDs, data,label
#Light Gradient Boosting Machine Feature Selection

def lgbmFeatureSelection(data_pd,fileName):
    IDs, X,y=get_data(data_pd)

    f=fileName
    print(f+"\n"+"Light gradient boosting machine learning for feateaure selection.....")
    model = LGBMClassifier(num_leaves=28,n_estimators=1024,max_depth=8,learning_rate=0.16,min_child_samples=28,random_state=2008,n_jobs=8)
    model.fit(X, y)
    importantFeatures = model.feature_importances_
    Values = np.sort(importantFeatures)[::-1]*0.618
    CriticalValue=np.mean(Values)
    K=importantFeatures.argsort()[::-1][:200]
    #K = importantFeatures.argsort()[::-1][:len(Values[Values>CriticalValue])]
    #if len(K) > 200:
    #LGB_ALL_K=pd.concat([y,X.iloc[:,A]],axis=1)
    #else:
    LGB_ALL_K=pd.concat([y,X.iloc[:,K]],axis=1)
    
    LGB_ALL_K.index=IDs
    LGB_ALL_K.to_csv(f)
    
    print("Selected Top %d features"%(LGB_ALL_K.shape[1]-1))
    print("LGBoosting features selections completed!!!!\n\n")
    
    return  LGB_ALL_K
    