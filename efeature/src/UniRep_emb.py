from __future__ import print_function,division

import sys
import os
sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
 
import time
 
from rich.progress import track
 
import numpy as np
import pandas as pd
import torch
import warnings
 
warnings.filterwarnings('ignore')
 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from tape import UniRepModel,TAPETokenizer

 
import preprocessing.fasta as fasta

def UniRep_Embed(fastaFile,outFile):
    T0=time.time()
    
    
     
     
    UNIREPEB_=[]
    
    inData=fasta.fasta2csv(fastaFile)
    SEQ_=inData["Seq"]
    PID_=inData["PID"]
    CLASS_=inData["Class"]
     
       
    
    print("UniRep Embedding...")
    
    
    model = UniRepModel.from_pretrained('babbler-1900')
    model=model.to(DEVICE)
    tokenizer = TAPETokenizer(vocab='unirep') 
    
     
     
    

    for sequence in track(SEQ_,"Computing: "):
    
             
        if len(sequence) == 0:
            print('# WARNING: sequence', pid, 'has length=0. Skipping.', file=sys.stderr)
            continue
          
        
        with torch.no_grad():
            token_ids = torch.tensor([tokenizer.encode(sequence)])
            token_ids = token_ids.to(DEVICE)
            output = model(token_ids)
            unirep_output = output[0]
            #print(unirep_output.shape)
            unirep_output=torch.squeeze(unirep_output)
            #print(unirep_output.shape)
            unirep_output= unirep_output.mean(0)
            unirep_output = unirep_output.cpu().numpy()
             
           # print(sequence,len(sequence),unirep_output.shape)
            UNIREPEB_.append(unirep_output.tolist())      
             
          
    unirep_feature=pd.DataFrame(UNIREPEB_)
     
    
    col=["UniRep_F"+str(i+1) for i in range(0,1900)]
    unirep_feature.columns=col
    unirep_feature=pd.concat([CLASS_,unirep_feature],axis=1)
    unirep_feature.index=PID_
    print(unirep_feature.shape)
    unirep_feature.to_csv(outFile)
    print("Getting Deep Representation Learning Features with UniRep is done.")
    print("it took %0.3f mins.\n"%((time.time()-T0)/60))

    return unirep_feature
    