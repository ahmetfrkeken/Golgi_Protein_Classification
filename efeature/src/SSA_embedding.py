from __future__ import print_function,division

import sys
import os
sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
 

 
from rich.progress import track
 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

from preprocessing.alphabets import Uniprot21
import preprocessing.fasta as fasta
 
import warnings
 
warnings.filterwarnings('ignore')

def unstack_lstm(lstm):
    device = next(iter(lstm.parameters())).device

    in_size = lstm.input_size
    hidden_dim = lstm.hidden_size
    layers = []
    for i in range(lstm.num_layers):
        layer = nn.LSTM(in_size, hidden_dim, batch_first=True, bidirectional=True)
        layer.to(device)

        attributes = ['weight_ih_l', 'weight_hh_l', 'bias_ih_l', 'bias_hh_l']
        for attr in attributes:
            dest = attr + '0'
            src = attr + str(i)
            getattr(layer, dest).data[:] = getattr(lstm, src)
             
            dest = attr + '0_reverse'
            src = attr + str(i) + '_reverse'
            getattr(layer, dest).data[:] = getattr(lstm, src)
             
        layer.flatten_parameters()
        layers.append(layer)
        in_size = 2*hidden_dim
    return layers

def embed_stack(x, lm_embed, lstm_stack, proj, include_lm=True, final_only=False):
    zs = []
    
    x_onehot = x.new(x.size(0),x.size(1), 21).float().zero_()
    x_onehot.scatter_(2,x.unsqueeze(2),1)
    zs.append(x_onehot)
    
    h = lm_embed(x)
    if include_lm and not final_only:
        zs.append(h)

    if lstm_stack is not None:
        for lstm in lstm_stack:
            h,_ = lstm(h)
            if not final_only:
                zs.append(h)
    if proj is not None:
        h = proj(h.squeeze(0)).unsqueeze(0)
        zs.append(h)

    z = torch.cat(zs, 2)
    return z


def embed_sequence(x, lm_embed, lstm_stack, proj, include_lm=True, final_only=False
                  ,  pool='none', use_cuda=False):

    if len(x) == 0:
        return None

    alphabet = Uniprot21()
    x = x.upper()
    # convert to alphabet index
    x = alphabet.encode(x)
    x = torch.from_numpy(x)
    if use_cuda:
        #x = x.cuda()
        x = x.to(DEVICE) 

    # embed the sequence
    with torch.no_grad():
        x = x.long().unsqueeze(0)
        z = embed_stack(x, lm_embed, lstm_stack, proj
                       , include_lm=include_lm, final_only=final_only)
        # pool if needed
        z = z.squeeze(0)
        if pool == 'sum':
            z = z.sum(0)
        elif pool == 'max':
            z,_ = z.max(0)
        elif pool == 'avg':
            z = z.mean(0)
        z = z.cpu().numpy()

    return z


def load_model(path, use_cuda=False):
    encoder = torch.load(path)
    #torch.save(encoder,"SA_emb_model.pk")
    #print(encoder)
     
    print("load_model...."+path)
    encoder.eval()

    #for name, parameter in encoder.named_parameters():
        #print(name, ':', parameter.size())

    if use_cuda:
        #encoder.cuda()
        encoder=encoder.to(DEVICE)

    #if type(encoder) is src.models.sequence.BiLM:
        ## model is only the LM
        #return encoder.encode, None, None

    encoder = encoder.embedding

    lm_embed = encoder.embed
    lstm_stack = unstack_lstm(encoder.rnn)
    proj = encoder.proj

    return lm_embed, lstm_stack, proj



def SSA_Embed(fastaFile,outFile):
     
    T0=time.time()
    SSAEMB_=[] 
    inData=fasta.fasta2csv(fastaFile)
    SEQ_=inData["Seq"]
    PID_=inData["PID"]
    CLASS_=inData["Class"]
    
  
     
     
    print("SSA Embedding...") 
    print("Loading SSA Model...",    file=sys.stderr, end='\r')
    lm_embed, lstm_stack, proj=load_model("./embbed_models/SSA_embed.model", use_cuda=False)
   
    include_lm=True
    final_only=True

     
    
     

    for sequence in track(SEQ_,"Computing: "):
        sequence = sequence.encode("utf-8")
            
        z = embed_sequence(sequence, lm_embed, lstm_stack, proj
                                  , include_lm=include_lm, final_only=final_only
                                  , pool='avg', use_cuda=True)
            
        SSAEMB_.append(z)
      #count += 1
            #print('#{}$'.format(count), file=sys.stderr, end='\r')
    
    
    ssa_feature=pd.DataFrame(SSAEMB_)
    col=["SSA_F"+str(i+1) for i in range(0,121)]
    ssa_feature.columns=col
    ssa_feature=pd.concat([CLASS_,ssa_feature],axis=1)
    ssa_feature.index=PID_
     
    print(ssa_feature.shape)
    ssa_feature.to_csv(outFile)
    print("SSA embedding finished@￥￥￥￥￥")
    print("it took %0.3f mins.\n"%((time.time()-T0)/60))

    return ssa_feature

def LM_Embed(fastaFile,outFile):
    T0=time.time()
    LMEMB_=[]
    inData=fasta.fasta2csv(fastaFile)
    SEQ_=inData["Seq"]
    PID_=inData["PID"]
    CLASS_=inData["Class"]
    
    
     
     
     
    print("Language Model(LM) Embedding...") 
    print("Loading LM Model...",    file=sys.stderr, end='\r')
    lm_embed, lstm_stack, proj=load_model("./embbed_models/SSA_embed.model", use_cuda=False)
    
    lm_only = True
    if lm_only:
        lstm_stack = None
        proj = None
   
   
    
     

    for sequence in track(SEQ_,"Computing: "):
        sequence = sequence.encode("utf-8")            
        z = embed_sequence(sequence, lm_embed, lstm_stack, proj
                                  ,  final_only=False,include_lm = True
                                  , pool='avg', use_cuda=True)
            
        LMEMB_.append(z)

    lm_feature=pd.DataFrame(LMEMB_)
    print(lm_feature.shape)
         
    
    
    col=["LM_F"+str(i+1) for i in range(0,533)]
    lm_feature.columns=col
    lm_feature=pd.concat([CLASS_,lm_feature],axis=1)
    lm_feature.index=PID_
    lm_feature.to_csv(outFile)

    print("LM embedding finished@@￥￥￥￥￥")
    print("it took %0.3f mins.\n"%((time.time()-T0)/60))

    return lm_feature

def BiLSTM_Embed(fastaFile,outFile):
    T0=time.time()
     
    BiLSTMEMB_=[] 
    inData=fasta.fasta2csv(fastaFile)
    SEQ_=inData["Seq"]
    PID_=inData["PID"]
    CLASS_=inData["Class"]
   
     
     
    print("\nBiLSTM Embedding...") 
    print("Loading BiLSTM Model...",    file=sys.stderr, end='\r')
    lm_embed, lstm_stack, proj=load_model("./embbed_models/SSA_embed.model", use_cuda=False)
    
    
    proj = None
   
    for sequence in track(SEQ_,"Computing: "):
        sequence = sequence.encode("utf-8")
            
        z = embed_sequence(sequence, lm_embed, lstm_stack, proj
                                  ,  final_only=False,include_lm = True
                                  , pool='avg', use_cuda=True)
            
        BiLSTMEMB_.append(z)
           
    
    
    bilstm_feature=pd.DataFrame(BiLSTMEMB_)
    print(bilstm_feature.shape)
    col=["BiLSTM_F"+str(i+1) for i in range(0,3605)]
    bilstm_feature.columns=col
    bilstm_feature=pd.concat([CLASS_,bilstm_feature],axis=1)
    bilstm_feature.index=PID_
    bilstm_feature.to_csv(outFile)
    print("BiLSTM embedding finished@@￥￥￥￥￥")
    print("it took %0.3f mins.\n"%((time.time()-T0)/60))

    return bilstm_feature
     





if __name__ == '__main__':
    import argparse
    import time
    parser = argparse.ArgumentParser('Script for embedding fasta format sequences using a saved embedding model. Saves embeddings as CSV file.')

    parser.add_argument('-i', help='sequences to embed in fasta format')
    parser.add_argument('-o', help='path to saved embedding CSV file')
    args = parser.parse_args()
    T0=time.time()
    SEQs=[]
    PIDs=[]

    with open(args.i, 'rb') as f:
        for name,sequence in fasta.parse_stream(f):
            #print(sequence)
            
            pid =str( name.decode('utf-8'))
            if len(sequence) == 0:
                print('# WARNING: sequence', pid, 'has length=0. Skipping.', file=sys.stderr)
                continue
            
            PIDs.append(pid)
            SEQs.append(sequence)

    SSA_Embed(SEQs,PIDs,args.o,use_cuda=False)
    print("It takes %0.3f mins.\n\n"%((time.time()-T0)/60))
    LM_Embed(SEQs,PIDs,args.o,use_cuda=False)
    print("It takes %0.3f mins.\n\n"%((time.time()-T0)/60))
    BiLSTM_Embed(SEQs,PIDs,args.o,use_cuda=False)
    print(" It takes %0.3f mins.\n\n"%((time.time()-T0)/60))
    
    
     
    




