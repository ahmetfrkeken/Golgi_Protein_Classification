import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.metrics import roc_curve,precision_recall_curve,make_scorer
from sklearn.metrics import precision_score,accuracy_score,auc, f1_score,recall_score,matthews_corrcoef,confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import confusion_matrix,classification_report


 



def ROC_PRC(y_true,y_proba,OutFile=None,outputROC=False):

### Get ROC curve
    FPR,TPR,thresholds_ =roc_curve(y_true, y_proba)
    
    sk_auROC=auc(FPR, TPR)

    ROC_pd=pd.DataFrame(np.zeros((len(FPR),2)), columns=["FPR","TPR"])
    ROC_pd["FPR"]=FPR
    ROC_pd["TPR"]=TPR
    #ROC_pd.to_csv(f+"_ROC.csv")
 
    mean_TPR = 0.0
    mean_FPR = np.linspace(0, 1, 100)
    mean_TPR += np.interp(mean_FPR, FPR, TPR)
    mean_TPR[0] = 0.0
    ROC_pd=pd.DataFrame(np.zeros((len(mean_FPR),2)), columns=["FPR","TPR"])
    ROC_pd["FPR"]=mean_FPR
    ROC_pd["TPR"]=mean_TPR

    if outputROC and (OutFile is not None):
        ROC_pd.to_csv(OutFile+"ROC_curve_intcepmean.csv")  

# get precision recall curve and auPRC
 
    PV, RV,th = precision_recall_curve(y_true, y_proba)
    sk_auPRC=auc(RV, PV)
    th=[]
    

    PRC_pd=pd.DataFrame(np.zeros((len(PV),2)), columns=["Recall","Precision"])
    PRC_pd["Recall"]=RV
    PRC_pd["Precision"]=PV
    #PRC_pd.to_csv(f+"PRC.csv")

    mean_PV = 0.0
    mean_RV = np.linspace(0, 1, 100)
    mean_PV += np.interp(mean_RV, RV, PV)
    mean_PV[0] = 0.0
    PRC_pd=pd.DataFrame(np.zeros((len(mean_RV),2)), columns=["Recall","Precision"])
    PRC_pd["Recall"]=mean_RV
    PRC_pd["Precision"]=mean_PV

    if outputROC and (OutFile is not None):
        PRC_pd.to_csv(OutFile+"_PRC_intcepmean.csv") 

    return sk_auROC,sk_auPRC

def CR2DF(ClassficationReport):
    CR=ClassficationReport
    idx=[]
    DF=pd.DataFrame()
    for k,subdict in CR.items():
        idx.append(k)
        tem=pd.DataFrame([subdict])
        DF=DF.append(tem) 
    DF.index=idx
    return DF

def getBinaryMetrics(y_true,y_pred,y_proba,OutFile=None,outputROC=False):
    ACC= accuracy_score(y_true,y_pred)
    MCC= matthews_corrcoef(y_true,y_pred)
    CM=confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = CM.ravel()
    Sn=tp/(tp+fn+1e-6)
    Sp=tn/(tn+fp+1e-6)

    F1= f1_score(y_true,y_pred)
    Precision=precision_score(y_true,y_pred)
    R_score=recall_score(y_true,y_pred)
    auROC,auPRC=ROC_PRC(y_true,y_proba,OutFile=None,outputROC=False)

    
    Results=np.array([ACC,MCC,Sn,Sp,auROC,auPRC,R_score,Precision,F1]).reshape(-1,9)
    #print(Results.shape)
    Metrics_=pd.DataFrame(Results,columns=["ACC","MCC","Sn","Sp","auROC","auPRC","Recall","Precision","F1"])
    print(Metrics_)

    return Metrics_

def getMultiClassMetrics(y_true,y_pred,y_proba,numclass=10,OutFile=None):
    ACC=accuracy_score(y_true,y_pred)
    precision=precision_score(y_true,y_pred,average='macro')
    recall=recall_score(y_true,y_pred,average='macro')
    F1=f1_score(y_true,y_pred,average='macro')
      
    #auROC=roc_auc_score(y_true,y_proba,average='macro',multi_class='ovr')
    kappa=cohen_kappa_score(y_true,y_pred)
    jaccard=jaccard_score(y_true,y_pred,average='macro')

    Results=np.array([ACC,precision,recall,F1,kappa,jaccard]).reshape(-1,6)
    Metrics_=pd.DataFrame(Results,columns=["ACC","Precision","F1","Recall","Kappa","Jaccard_score"])
    print(Metrics_)

    CM=confusion_matrix(y_true,y_pred)
    CR=classification_report(y_true,y_pred,output_dict=True)
    CM=pd.DataFrame(CM,index=range(0,numclass),columns=range(0,numclass)) 
    CM['pred_sum'] = CM.apply(lambda x: x.sum(), axis=1)
    print(CM)
    
    CR=classification_report(y_true,y_pred,output_dict=True)

    CR=CR2DF(CR)
    print(CR)
    

    return Metrics_,CM,CR



    
     