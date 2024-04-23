
from imblearn.over_sampling import SMOTE , ADASYN
import pandas as pd
smo=SMOTE()
def getSMOTE(data,outFile):
    X=data.iloc[:,1:].values
    y=data.iloc[:,0].values
    Xsmo,ysmo=smo.fit_resample(X,y)
    Xsmo=pd.DataFrame(Xsmo,columns=data.columns[1:])
    ysmo=pd.DataFrame(ysmo,columns=["Class"])
    data_smote=pd.concat([ysmo,Xsmo],axis=1)
    data_smote.to_csv(outFile)
    print(outFile,data_smote.shape)
    return data_smote