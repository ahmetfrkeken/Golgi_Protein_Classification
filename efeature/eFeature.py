from __future__ import print_function,division
import argparse
import time
from src.SSA_embedding import SSA_Embed,LM_Embed,BiLSTM_Embed
from src.W2V_emb import W2V_embbed01 
from src.UniRep_emb import UniRep_Embed
from src.TAPE_emb import Bert_Embed
from preprocessing  import fasta
from preprocessing.FeatureSelection import lgbmFeatureSelection
from preprocessing.PerformClassification import performBinaryClassifiers, performMulticlassClassifiers
from preprocessing.UMAP import plotUMAP
from preprocessing.SMOTE import getSMOTE
import shutil, os
import pandas as pd
import numpy as np
from rich.progress import track


def del_file(filepath):
    del_list = os.listdir(filepath)
    for f in track(del_list,"initializing..."):
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


FeatureName=["SSA","LM","BiLSTM","UniRep","TAPE_BERT","W2V","FusedAll"]

#FeatureName=["TAPE_BERT","FusedAll"]


if not os.path.exists("./out"):
    os.mkdir("./out")
else:
    del_file("./out")

for fname in track(FeatureName,"initializing..."):
    if not os.path.exists("./out/"+fname):
        os.mkdir("./out/"+fname)

    if not os.path.exists("./out/"+fname+"/Results"):
        os.mkdir("./out/"+fname+"/Results")

def GnerateFeatures(infasta,FeatureName,outcsv):
    
    T0=time.time()
    FeatureName=FeatureName

    if FeatureName=="SSA":   
        features = SSA_Embed(infasta, outcsv)
        print("SSA:", features.shape)

    elif FeatureName=="LM":   
        features = LM_Embed(infasta, outcsv)
        print("LM:", features.shape)
    
    elif FeatureName=="BiLSTM":   
        features = BiLSTM_Embed(infasta, outcsv)
        print("BiLSTM:", features.shape)

    elif FeatureName=="UniRep":   
        features = UniRep_Embed(infasta, outcsv)
        print("UniRep:", features.shape)
    
    elif FeatureName=="TAPE_BERT":   
        features = Bert_Embed(infasta, outcsv)
        print("TAPE_BERT:", features.shape)
    
    elif FeatureName=="W2V":   
        features = W2V_embbed01(infasta, outcsv)
        print("Word2Vec:", features.shape)
    
    print("%s embeding time consumed is %0.3f mins passed.\n\n"%(FeatureName,(time.time()-T0)/60))
    
    return features

 



def gsFeature(inTrainFasta,inTestFasta=None,outcsv="out_train.csv",smoteFlag=0):
    ## generate features and do feature selection
    fusedAllFeature_train = pd.DataFrame()
    train_ID_class=pd.DataFrame()
    train_dim=0
   

    fusedAllFeature_test = pd.DataFrame()
    test_ID_class=pd.DataFrame()
    test_dim=0
    

    
    for fname in FeatureName[:-1]:

        print("\n"+chr(10059)*28+fname+chr(10059)*28)
        ###generate TrainData features
        outFile="./out/"+fname+"/"+fname+"_"+outcsv.replace(".csv","")+"_Train.csv"
        trainFeature = GnerateFeatures(inTrainFasta,fname,outFile)
        umap_Figure=plotUMAP(trainFeature, outPNG=outFile.replace(".csv","")+"_UMAP")

        #fuse trainData features
        fusedAllFeature_train=pd.concat([fusedAllFeature_train,trainFeature.iloc[:,1:]],axis=1)
        if train_dim==0:
            train_ID_class["Class"]=trainFeature["Class"]
            train_ID_class["PID"]=trainFeature.index
        
        train_dim=train_dim+(trainFeature.shape[1]-1)
        print("train_dim=",train_dim)

        ##select trainData features
        outFile= outFile.replace("_Train.csv","_Train_lgbmSF.csv")
        select_trainFeatures=lgbmFeatureSelection(trainFeature,outFile)
        umap_SFigure=plotUMAP(select_trainFeatures,outPNG=outFile.replace(".csv","")+"_UMAP")

        

        if inTestFasta is not None:
            print("\n"+chr(931)*28+fname+chr(931)*28)
            #generate Testdata features
            outFile= "./out/"+fname+"/"+fname+"_"+outcsv.replace(".csv","")+"_Test.csv"
            testFeature = GnerateFeatures(inTestFasta,fname,outFile)
            testFeature.to_csv(outFile)
            umap_Figure=plotUMAP(testFeature,outPNG=outFile.replace(".csv","_UMAP"))
            
            ##fuse TestData Feature
            fusedAllFeature_test=pd.concat([fusedAllFeature_test,testFeature.iloc[:,1:]],axis=1)
            if test_dim==0:
                test_ID_class["Class"]=testFeature["Class"]
                test_ID_class["PID"]=testFeature.index
            test_dim=test_dim+(testFeature.shape[1]-1)
            print("test_dim=",test_dim)

            
            ## use selected trainData feature sto select testData features
            select_testFeatures=testFeature[select_trainFeatures.columns]
            select_testFeatures.index=testFeature.index
            outFile= outFile.replace("_Test.csv","_Test_lgbmSF.csv")
            select_testFeatures.to_csv(outFile)
            umap_SFigure=plotUMAP(select_testFeatures,outPNG=outFile.replace(".csv","_UMAP"))

        
        if smoteFlag==1:
            print("\n"+chr(10059)*28+"SMOTE"+chr(10059)*28)
            ###smote traindata 
            outFile="./out/"+fname+"/"+fname+"_"+outcsv.replace(".csv","")+"_Train_SMOTE.csv"
            smote_train_features=getSMOTE(trainFeature,outFile)
            umap_Figure=plotUMAP(smote_train_features, outPNG=outFile.replace(".csv","")+"_UMAP")

            ## selected features from smoted trainData
            outFile= outFile.replace(".csv","_lgbmSF.csv")
            select_trainFeatures=lgbmFeatureSelection(smote_train_features,outFile)
            umap_Figure=plotUMAP(select_trainFeatures, outPNG=outFile.replace(".csv","")+"_UMAP")
            if inTestFasta is not None:
            ## selected  test features from selected and smoted trainData features
                outFile="./out/"+fname+"/"+fname+"_"+outcsv.replace(".csv","")+"_Test_SMOTE_lgbmSF.csv"
                select_testFeatures=testFeature[select_trainFeatures.columns]
                select_testFeatures.index=testFeature.index
                select_testFeatures.to_csv(outFile)
                umap_SFigure=plotUMAP(select_testFeatures,outPNG=outFile.replace(".csv","_UMAP"))



    ###All features fusion 
     
    fname=FeatureName[-1]
    print("\n"+chr(10059)*28+fname+chr(10059)*28)
    print(fusedAllFeature_train.shape,str(train_dim))
    fusedAllFeature_train=pd.concat([train_ID_class["Class"] ,fusedAllFeature_train],axis=1)
    fusedAllFeature_train.index= train_ID_class["PID"]
    print(fusedAllFeature_train.shape)

    outFile="./out/"+fname+"/"+fname+"_"+outcsv.replace(".csv","")+"_Train.csv"
    fusedAllFeature_train.to_csv(outFile)
    umap_Figure=plotUMAP(fusedAllFeature_train, outPNG=outFile.replace(".csv","")+"_UMAP")
    outFile= outFile.replace("_Train.csv","_Train_lgbmSF.csv")
    select_trainFeatures=lgbmFeatureSelection(fusedAllFeature_train,outFile)
    umap_SFigure=plotUMAP(select_trainFeatures,outPNG=outFile.replace(".csv","")+"_UMAP")

    if inTestFasta is not None:
        fname=FeatureName[-1]
        print("\n"+chr(931)*28+fname+chr(931)*28)
        print(fusedAllFeature_test.shape,str(test_dim))
        fusedAllFeature_test=pd.concat([test_ID_class["Class"] ,fusedAllFeature_test],axis=1)
        fusedAllFeature_test.index= test_ID_class["PID"]
        print(fusedAllFeature_test.shape)
        outFile="./out/"+fname+"/"+fname+"_"+outcsv.replace(".csv","")+"_Test.csv"
        fusedAllFeature_test.to_csv(outFile)
        umap_Figure=plotUMAP(fusedAllFeature_test, outPNG=outFile.replace(".csv","")+"_UMAP")

        outFile= outFile.replace("_Test.csv","_Test_lgbmSF.csv")
        select_testFeatures=fusedAllFeature_test[select_trainFeatures.columns]
            #select_testFeatures=pd.concat([testFeature.iloc[:,0],select_testFeatures],axis=1)
        select_testFeatures.index=fusedAllFeature_test.index
        select_testFeatures.to_csv(outFile)
        umap_SFigure=plotUMAP(select_testFeatures,outPNG=outFile.replace(".csv","_UMAP"))

    if smoteFlag==1:
        print("\n"+chr(10059)*28+"SMOTE"+chr(10059)*28)
        outFile="./out/"+fname+"/"+fname+"_"+outcsv.replace(".csv","")+"_Train_SMOTE.csv"
        smote_fusedAllFeatures_train=getSMOTE(fusedAllFeature_train,outFile)
        umap_Figure=plotUMAP(smote_fusedAllFeatures_train, outPNG=outFile.replace(".csv","")+"_UMAP")
        outFile= outFile.replace(".csv","_lgbmSF.csv")
        select_trainFeatures=lgbmFeatureSelection(smote_fusedAllFeatures_train,outFile)
        umap_SFigure=plotUMAP(select_trainFeatures,outPNG=outFile.replace(".csv","_UMAP"))

        if inTestFasta is not None:
            ## selected  test features from selected and smoted trainData features
                outFile="./out/"+fname+"/"+fname+"_"+outcsv.replace(".csv","")+"_Test_SMOTE_lgbmSF.csv"
                select_testFeatures=fusedAllFeature_test[select_trainFeatures.columns]
                select_testFeatures.index=fusedAllFeature_test.index
                select_testFeatures.to_csv(outFile)
                umap_SFigure=plotUMAP(select_testFeatures,outPNG=outFile.replace(".csv","_UMAP"))
             
    

   


    print("Feature extraction done!!!")

def run(inTrainFasta,inTestFasta,outFile,smoteFlag,runMode,numclass,kfold):
    
    if runMode==0 : ## get sequence embedding features only
        
        gsFeature(inTrainFasta,inTestFasta,outFile,smoteFlag)


    elif runMode==1: #get sequence embedding features and do binary classification
        gsFeature(inTrainFasta,inTestFasta,outFile,smoteFlag)
        
        for fname in track(FeatureName,"reading Data..."):

            ###orignal features for training, validation or testing 
            trainData="./out/"+fname+"/"+fname+"_"+outFile.replace(".csv","")+"_Train.csv"
            testData=None
            if inTestFasta is not None:
                testData="./out/"+fname+"/"+fname+"_"+outFile.replace(".csv","")+"_Test.csv"

            outPath="./out/"+fname+"/Results/ValANDTest"
            if not os.path.exists(outPath):
                os.mkdir(outPath)
            
            performBinaryClassifiers(trainData,testData,featureName=fname,outPath=outPath,kfold=kfold)


            ###selected features for training, validation or testing
            trainData="./out/"+fname+"/"+fname+"_"+outFile.replace(".csv","")+"_Train_lgbmSF.csv"
            testData=None
            if inTestFasta is not None:
                testData="./out/"+fname+"/"+fname+"_"+outFile.replace(".csv","")+"_Test_lgbmSF.csv"

            outPath="./out/"+fname+"/Results/ValANDTest_SelectedFeatures"
            if not os.path.exists(outPath):
                os.mkdir(outPath)
            
            performBinaryClassifiers(trainData,testData,featureName=fname,outPath=outPath,kfold=kfold)



            if smoteFlag:
                ###orignal features for training, validation or testing after SMOTE
                trainData="./out/"+fname+"/"+fname+"_"+outFile.replace(".csv","")+"_Train_SMOTE.csv"
                testData=None
                if inTestFasta is not None:
                    testData="./out/"+fname+"/"+fname+"_"+outFile.replace(".csv","")+"_Test.csv"

                outPath="./out/"+fname+"/Results/SMOTE_ValANDTest"
                if not os.path.exists(outPath):
                    os.mkdir(outPath)
            
                performBinaryClassifiers(trainData,testData,featureName=fname,outPath=outPath,kfold=kfold)


                ###Selected features for training, validation or testing 

                trainData="./out/"+fname+"/"+fname+"_"+outFile.replace(".csv","")+"_Train_SMOTE_lgbmSF.csv"
                testData=None
                if inTestFasta is not None:
                    testData="./out/"+fname+"/"+fname+"_"+outFile.replace(".csv","")+"_Test_SMOTE_lgbmSF.csv"

                outPath="./out/"+fname+"/Results/SMOTE_lgbmSF_ValANDTest"
                if not os.path.exists(outPath):
                    os.mkdir(outPath)
            
                performBinaryClassifiers(trainData,testData,featureName=fname,outPath=outPath,kfold=kfold)


             

    elif runMode==2: #get sequence embedding features and do multiclass classification
        
        
        gsFeature(inTrainFasta,inTestFasta,outFile,smoteFlag)        
        

        for fname in track(FeatureName,"reading Data..."):

            ###orignal features for training, validation or testing 
            trainData="./out/"+fname+"/"+fname+"_"+outFile.replace(".csv","")+"_Train.csv"
            testData=None
            if inTestFasta is not None:
                testData="./out/"+fname+"/"+fname+"_"+outFile.replace(".csv","")+"_Test.csv"

            outPath="./out/"+fname+"/Results/ValANDTest"
            if not os.path.exists(outPath):
                os.mkdir(outPath)
            
            performMulticlassClassifiers(trainData,testData,featureName=fname,outPath=outPath,kfold=kfold,numclass=numclass)


            ###selected features for training, validation or testing
            trainData="./out/"+fname+"/"+fname+"_"+outFile.replace(".csv","")+"_Train_lgbmSF.csv"
            testData=None
            if inTestFasta is not None:
                testData="./out/"+fname+"/"+fname+"_"+outFile.replace(".csv","")+"_Test_lgbmSF.csv"

            outPath="./out/"+fname+"/Results/ValANDTest_SelectedFeatures"
            if not os.path.exists(outPath):
                os.mkdir(outPath)
            
            performMulticlassClassifiers(trainData,testData,featureName=fname,outPath=outPath,kfold=kfold,numclass=numclass)



            if smoteFlag:
                ###orignal features for training, validation or testing after SMOTE
                trainData="./out/"+fname+"/"+fname+"_"+outFile.replace(".csv","")+"_Train_SMOTE.csv"
                testData=None
                if inTestFasta is not None:
                    testData="./out/"+fname+"/"+fname+"_"+outFile.replace(".csv","")+"_Test.csv"

                outPath="./out/"+fname+"/Results/SMOTE_ValANDTest"
                if not os.path.exists(outPath):
                    os.mkdir(outPath)
            
                performMulticlassClassifiers(trainData,testData,featureName=fname,outPath=outPath,kfold=kfold,numclass=numclass)


                ###Selected features for training, validation or testing 

                trainData="./out/"+fname+"/"+fname+"_"+outFile.replace(".csv","")+"_Train_SMOTE_lgbmSF.csv"
                testData=None
                if inTestFasta is not None:
                    testData="./out/"+fname+"/"+fname+"_"+outFile.replace(".csv","")+"_Test_SMOTE_lgbmSF.csv"

                outPath="./out/"+fname+"/Results/SMOTE_lgbmSF_ValANDTest"
                if not os.path.exists(outPath):
                    os.mkdir(outPath)
            
                performMulticlassClassifiers(trainData,testData,featureName=fname,outPath=outPath,kfold=kfold,numclass=numclass)





   


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('Script for embedding fasta format sequences using a pretained  deep representaion learning  model. Saves embeddings feature as CSV file.')

    parser.add_argument('--inTrain', type=str, help='sequences for TRAINing to embed in fasta format')
    parser.add_argument('--inTest',type=str, default=None,help='sequences for Testing to embed in fasta format')
    parser.add_argument('--out', type=str, default="eFeature_out.csv",help='path to saved embedding CSV file')
    parser.add_argument('--smote', type=int, default=1, help="0: no data SMOTE, 1: data SMOTE")
    parser.add_argument('--mode', type=int, default=0, help="mode=0:only generate features and do feature selection;\n mode=1: do feature extraction and selection then do binaryclassification;\n \n mode=2: do feature extraction and selection then do multiclass-classification")
    
     
    parser.add_argument('--numclass', type=int, default=2, help="2: binary classes, 10: ten classes, etc.")
    parser.add_argument('--kfold', type=int, default=5, help="kfold=5: five-fold cross-validation")

    args = parser.parse_args()

    inTrainFasta=args.inTrain
    inTestFasta=args.inTest
    outFile=args.out
    
    smoteFlag=args.smote
    runMode=args.mode
     
    numclass=args.numclass
    kfold=args.kfold


    T0=time.strftime ( "%H:%M:%S" )    
    run(inTrainFasta,inTestFasta,outFile,smoteFlag,runMode,numclass,kfold)
    Te=time. strftime ("%H:%M:%S" )
    print("\nStart Time:",T0,"\nStop Time:",Te)
   
    
     
