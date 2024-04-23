# Avoiding warning
import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn
# _______________________________

#
import pandas as pd
import numpy as np
from joblib import dump,load
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  RandomForestClassifier,  AdaBoostClassifier, ExtraTreesClassifier
from lightgbm.sklearn import LGBMClassifier 
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold
from preprocessing.skMetrics import getBinaryMetrics, getMultiClassMetrics


ClassifierNames = ['KNN','LR', 'GNB', "SVM", 'RF', 'LGBM']
Classifiers = [
    KNeighborsClassifier(n_neighbors=5), #KNN
    #LinearDiscriminantAnalysis(),#LDA
    LogisticRegression(penalty='l2', C=10, max_iter=500, solver='sag'),   #LR  
    GaussianNB(), #'GNB'     
    SVC(kernel='rbf',   probability=True), #SVM   
    #DecisionTreeClassifier(),#DT               
    RandomForestClassifier(),  #RF    
    #AdaBoostClassifier(),#ADB
    #ExtraTreesClassifier(),#ET
    LGBMClassifier()#LGBM
]

 


def performBinaryClassifiers(trainData,testData=None,featureName="SSA",outPath="./",kfold=10):

    print("Load Train Data...")
    print(trainData)

    D = pd.read_csv(trainData,index_col=0,header=0)
    X = D.iloc[:,1:]
    y = D.iloc[:, 0].values
    print(trainData)
    print(X.shape,y.shape)
    

    x_t=[]
    y_t=[]
    if testData is not None:
        print("Load Test Data...")
        print(testData)
        T = pd.read_csv(testData,index_col=0,header=0)
        x_t=T[X.columns].values
        y_t=T.iloc[:,0].values
        print(testData)
        print(x_t.shape,y_t.shape)
    
    X=X.values
    
      

    scale = StandardScaler()
    X = scale.fit_transform(X)
    X_t= scale.transform(x_t)
    dump(scale,outPath+featureName+"_trainData.StandardScaler.model")
         
    col_name=["ACC","MCC","Sn","Sp","auROC","auPRC","Recall","Precision","F1"]  
    cv = StratifiedKFold(n_splits=kfold, shuffle=True)
    eachFoldResults=pd.DataFrame()
    ValMeanResults=pd.DataFrame()
    TestMeanResults=pd.DataFrame()
    Results=pd.DataFrame()

    for classifier, name in zip(Classifiers, ClassifierNames):
        print('{} is done.'.format(classifier.__class__.__name__))
        
        
        model = classifier
        fold_count=0
        temTrainRpd=pd.DataFrame()##store each classifer results for each fold
        temTestRpd=pd.DataFrame()#
         

        for (train_index, test_index) in cv.split(X, y):
            
            fold_count=fold_count+1
            print("Fold"+str(fold_count)+"__Validataion Results___\n")

            X_train = X[train_index]
            X_test = X[test_index]

            y_train = y[train_index]
            y_test = y[test_index]

            
            model.fit(X_train, y_train)
                       
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred_label = model.predict(X_test)
            
            each_fold_metrics_valid=getBinaryMetrics(y_test,y_pred_label,y_pred_proba)
            each_fold_metrics_valid.columns=["Val_"+ m for m in col_name]

            
           
            temTrainRpd=temTrainRpd.append(each_fold_metrics_valid,ignore_index=True)


            if testData is not None:
                #getBinaryMetrics(y_true,y_pred,y_proba,OutFile=None,outputROC=False):
                print("Fold"+str(fold_count)+"__Independent test___\n")
                yt_pred_proba=model.predict_proba(X_t)[:,1]
                yt_pred_label=model.predict(X_t)
                each_fold_metrics_test=getBinaryMetrics(y_t, yt_pred_label,yt_pred_proba)
                each_fold_metrics_test.columns=["Test_"+ m for m in col_name]

                temTestRpd=temTestRpd.append(each_fold_metrics_test,ignore_index=True)
        
        col_Name_Feature_fold={
            "Classifier":[name for i in range(fold_count)],
            "Feature": [featureName for i in range(fold_count)],
            "KFold": [fold_count for i in range(fold_count)]
        }

        col_Name_Feature_fold=pd.DataFrame(col_Name_Feature_fold,columns=["Classifier","Feature","KFold" ])
        print(col_Name_Feature_fold)
            
        print(temTrainRpd.mean)
        ValMeanResults[name+"_"+featureName]=temTrainRpd.mean()
        print(ValMeanResults)
        if  testData is not None:
            print(temTestRpd.mean)
            TestMeanResults[name+"_"+featureName]=temTestRpd.mean()
            print(TestMeanResults)
        

        
        eachFoldResults=pd.concat([col_Name_Feature_fold,temTrainRpd, temTestRpd],axis=1,ignore_index=True)
        print(eachFoldResults)
        Results=Results.append(eachFoldResults,ignore_index=True)

    

    print(ValMeanResults)
    if  testData is not None:
        print(TestMeanResults)
    
    if testData is not None:
    #"ACC","MCC","Sn","Sp","auROC","auPRC","Recall","Precision","F1"
        Results.columns=["Method","Feature","Fold","validation_ACC","validation_MCC","validation_Sn","validation_Sp","validation_auROC","validation_auPRC",\
        "validation_Recall","validation_Precision","validation_F1","test_ACC","test_MCC","test_Sn","test_Sp",\
        "test_auROC","test_auPRC","test_Recall","test_Precision","test_F1"]
    else:
        Results.columns=["Method","Feature","Fold","validation_ACC","validation_MCC","validation_Sn","validation_Sp","validation_auROC","validation_auPRC",\
        "validation_Recall","validation_Precision","validation_F1"]

    print(Results)

    if  testData is not None:
        
        Results.to_csv(outPath+"/EachFold_Validation_Test_Results.csv")
        ValMeanResults=pd.DataFrame(ValMeanResults.values.T, index=ValMeanResults.columns, columns=ValMeanResults.index)
        TestMeanResults=pd.DataFrame(TestMeanResults.values.T, index=TestMeanResults.columns, columns=TestMeanResults.index)

        ValMeanResults.to_csv(outPath+"/Validation_MeanResults.csv")
        TestMeanResults.to_csv(outPath+"/Test_MeanResults.csv")
        return  Results,ValMeanResults,TestMeanResults
    else:
        Results.to_csv(outPath+"/EachFold_Validation_Results.csv")
        ValMeanResults=pd.DataFrame(ValMeanResults.values.T, index=ValMeanResults.columns, columns=ValMeanResults.index)
        ValMeanResults.to_csv(outPath+"/Validation_MeanResults.csv")
        return  Results,ValMeanResults
                
                

              

def performMulticlassClassifiers(trainData,testData=None,featureName="SSA",outPath="./",kfold=10,numclass=10):

    print("Load Train Data...")
    print(trainData)

    D = pd.read_csv(trainData,index_col=0,header=0)
    X = D.iloc[:,1:]
    y = D.iloc[:, 0].values
    print(trainData)
    print(X.shape,y.shape)
    

    x_t=[]
    y_t=[]
    if testData is not None:
        print("Load Test Data...")
        print(testData)
        T = pd.read_csv(testData,index_col=0,header=0)
        x_t=T[X.columns].values
        y_t=T.iloc[:,0].values
        print(testData)
        print(x_t.shape,y_t.shape)
    
    X=X.values
    
      

    scale = StandardScaler()
    X = scale.fit_transform(X)
    X_t= scale.transform(x_t)
    dump(scale,outPath+featureName+"_trainData.StandardScaler.model")
         
    col_name=["ACC","Precision","F1","Recall","Kappa","Jaccard_score"]  
    cv = StratifiedKFold(n_splits=kfold, shuffle=True)
    eachFoldResults=pd.DataFrame()
    ValMeanResults=pd.DataFrame()
    TestMeanResults=pd.DataFrame()
    Results=pd.DataFrame()

    for classifier, name in zip(Classifiers, ClassifierNames):
        print('{} is done.'.format(classifier.__class__.__name__))
        
        
        model = classifier
        fold_count=0
        temTrainRpd=pd.DataFrame()##store each classifer results for each fold
        temTestRpd=pd.DataFrame()#
         

        for (train_index, test_index) in cv.split(X, y):
            
            fold_count=fold_count+1
            print("Fold"+str(fold_count)+"__Validataion Results___\n")

            X_train = X[train_index]
            X_test = X[test_index]

            y_train = y[train_index]
            y_test = y[test_index]

            
            model.fit(X_train, y_train)
                       
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred_label = model.predict(X_test)
            
            each_fold_metrics_valid,each_CM,each_CR=getMultiClassMetrics(y_test,y_pred_label,y_pred_proba,numclass=numclass)
            each_fold_metrics_valid.columns=["Val_"+ m for m in col_name]
            each_CM.to_csv(outPath+"/"+name+"_Fold"+str(fold_count)+"_Validataion Results_ConfusionMatrix.csv")
            each_CR.to_csv(outPath+"/"+name+"_Fold"+str(fold_count)+"_Validataion Results_ClassificationReport.csv")



            
           
            temTrainRpd=temTrainRpd.append(each_fold_metrics_valid,ignore_index=True)


            if testData is not None:
                #getMultiClassMetrics(y_test,y_pred_label,y_pred_proba,numclass=numclass)
                print("Fold"+str(fold_count)+"__Independent test___\n")
                yt_pred_proba=model.predict_proba(X_t)[:,1]
                yt_pred_label=model.predict(X_t)
                each_fold_metrics_test,each_CM,each_CR=getMultiClassMetrics(y_t, yt_pred_label,yt_pred_proba,numclass=numclass)
                each_CM.to_csv(outPath+"/"+name+"_Fold"+str(fold_count)+"_Independent test_ConfusionMatrix.csv")
                each_CR.to_csv(outPath+"/"+name+"_Fold"+str(fold_count)+"_Independent test_ClassificationReport.csv")

                each_fold_metrics_test.columns=["Test_"+ m for m in col_name]

                temTestRpd=temTestRpd.append(each_fold_metrics_test,ignore_index=True)
        
        col_Name_Feature_fold={
            "Classifier":[name for i in range(fold_count)],
            "Feature": [featureName for i in range(fold_count)],
            "KFold": [fold_count for i in range(fold_count)]
        }

        col_Name_Feature_fold=pd.DataFrame(col_Name_Feature_fold,columns=["Classifier","Feature","KFold" ])
        print(col_Name_Feature_fold)
            
        print(temTrainRpd.mean)
        ValMeanResults[name+"_"+featureName]=temTrainRpd.mean()
        print(ValMeanResults)
        if  testData is not None:
            print(temTestRpd.mean)
            TestMeanResults[name+"_"+featureName]=temTestRpd.mean()
            print(TestMeanResults)
        

         
        eachFoldResults=pd.concat([col_Name_Feature_fold,temTrainRpd, temTestRpd],axis=1,ignore_index=True)
        print(eachFoldResults)
        Results=Results.append(eachFoldResults,ignore_index=True)

         
    print(ValMeanResults)
    if  testData is not None:
        print(TestMeanResults)

    if testData is not none:
        Results.columns=["Method","Feature","Fold","validation_ACC","validation_Precision","validation_F1","validation_Recall","validation_Kappa","validation_Jaccard_score",\
        "test_ACC","test_Precision","test_F1","Recall","test_Kappa","test_Jaccard_score"]
    else:
        Results.columns=["Method","Feature","Fold","validation_ACC","validation_Precision","validation_F1","validation_Recall","validation_Kappa","validation_Jaccard_score"]
        
    print(Results)

    if  testData is not None:
        
        Results.to_csv(outPath+"/EachFold_Validation_Test_Results.csv")
        ValMeanResults=pd.DataFrame(ValMeanResults.values.T, index=ValMeanResults.columns, columns=ValMeanResults.index)
        TestMeanResults=pd.DataFrame(TestMeanResults.values.T, index=TestMeanResults.columns, columns=TestMeanResults.index)

        ValMeanResults.to_csv(outPath+"/Validation_MeanResults.csv")
        TestMeanResults.to_csv(outPath+"/Test_MeanResults.csv")
        return  Results,ValMeanResults,TestMeanResults
    else:
        Results.to_csv(outPath+"/EachFold_Validation_Results.csv")
        ValMeanResults=pd.DataFrame(ValMeanResults.values.T, index=ValMeanResults.columns, columns=ValMeanResults.index)
        ValMeanResults.to_csv(outPath+"/Validation_MeanResults.csv")
        return  Results,ValMeanResults    
            

            
           
            
             
            
            
            
            
            

      



