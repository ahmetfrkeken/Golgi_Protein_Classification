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
print(DEVICE)

from tape import ProteinBertModel,TAPETokenizer

 
import preprocessing.fasta as fasta

def Bert_Embed(fastaFile,outFile):
    T0=time.time()
    TAPEEMB_=[]
    
    inData=fasta.fasta2csv(fastaFile)
    SEQ_=inData["Seq"]
    PID_=inData["PID"]
    CLASS_=inData["Class"]
     
       
    
    print("Bert-based Embedding...")
    
    
    model = ProteinBertModel.from_pretrained('bert-base')
    model=model.to(DEVICE)
    tokenizer = TAPETokenizer(vocab='iupac')
    
     
     
    

    for sequence in track(SEQ_,"Computing: "):
    
             
        if len(sequence) == 0:
            print('# WARNING: sequence', pid, 'has length=0. Skipping.', file=sys.stderr)
            continue
          
        
        with torch.no_grad():
            token_ids = torch.tensor([tokenizer.encode(sequence)])
            token_ids = token_ids.to(DEVICE)
            output = model(token_ids)
            bert_output = output[0]
            #print(bert_output.shape)
            bert_output=torch.squeeze(bert_output)
            #print(bert_output.shape)
            bert_output= bert_output.mean(0)
            bert_output = bert_output.cpu().numpy()
             
            #print(len(sequence),bert_output.shape)
            TAPEEMB_.append(bert_output.tolist())      
             
          
    bert_feature=pd.DataFrame(TAPEEMB_)
     
    
    col=["TAPE_BERT_F"+str(i+1) for i in range(0,768)]
    bert_feature.columns=col
    bert_feature=pd.concat([CLASS_,bert_feature],axis=1)
    bert_feature.index=PID_
    bert_feature.to_csv(outFile)
    print("Getting Deep Representation Learning Features with bert based model is done.")
    print("it took %0.3f mins.\n"%((time.time()-T0)/60))
    
    return bert_feature