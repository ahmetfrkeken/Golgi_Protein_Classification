from gensim.models import word2vec
from preprocessing.W2Vutils import *
import sys

import random
AA='ARNDCQEGHILKMFPSTWYV'
 
AA=list(AA)
 

 
from rich.progress import track
 
import numpy as np
import pandas as pd
from preprocessing  import fasta
import time
def load_protvec(model_fname):
    return word2vec.Word2Vec.load(model_fname)


class ProtVec(word2vec.Word2Vec):

    def __init__(self, fasta_fname=None, corpus=None, n=3, size=100, corpus_fname="corpus.txt",  sg=1, window=25, min_count=1, workers=3):
        """
        Either fname or corpus is required.

        fasta_fname: fasta file for corpus
        corpus: corpus object implemented by gensim
        n: n of n-gram
        corpus_fname: corpus file path
        min_count: least appearance count in corpus. if the n-gram appear k times which is below min_count, the model does not remember the n-gram
        """

        self.n = n
        self.size = size
        self.fasta_fname = fasta_fname

        if corpus is None and fasta_fname is None:
            raise Exception("Either fasta_fname or corpus is needed!")

        if fasta_fname is not None:
            print('Generate Corpus file from fasta file...')
            generate_corpusfile(fasta_fname, n, corpus_fname)
            corpus = word2vec.Text8Corpus(corpus_fname)

        word2vec.Word2Vec.__init__(self, corpus, size=size, sg=sg, window=window, min_count=min_count, workers=workers)

    def to_vecs(self, seq):
        """
        convert sequence to three n-length vectors
        e.g. 'AGAMQSASM' => [ array([  ... * 100 ], array([  ... * 100 ], array([  ... * 100 ] ]
        """
        ngram_patterns = split_ngrams(seq, self.n)

        protvecs = []
        for ngrams in ngram_patterns:
            ngram_vecs = []
            for ngram in ngrams:
                try:
                    ngram_vecs.append(self[ngram])
                except:
                    raise Exception("Model has never trained this n-gram: " + ngram)
            protvecs.append(sum(ngram_vecs))
        return protvecs

def W2V_embbed01(inFasta,outFile):

    print("Word2Vec Embedding...")
    T0=time.time()
    PV=load_protvec("./embbed_models/W2V_model01.bin")
    W2VEMB_=[] 
    inData=fasta.fasta2csv(inFasta)
    SEQ_=inData["Seq"]
    
    PID_=inData["PID"]
    CLASS_=inData["Class"]

    for sequence in track(SEQ_,"Computing: "):
         
        Xrep=AA[random.randint(0,len(AA)-1)]
        sequence=sequence.replace("X",Xrep)
        Xrep=AA[random.randint(0,len(AA)-1)]
        sequence=sequence.replace("B",Xrep)
        Xrep=AA[random.randint(0,len(AA)-1)]
        sequence=sequence.replace("U",Xrep)
        Xrep=AA[random.randint(0,len(AA)-1)]
        sequence=sequence.replace("Z",Xrep)

        z =  PV.to_vecs(sequence)
        for i in range(3):
           if np.array(z[i]).shape!=(100,):
               z[i]=[0 for j in range(100)]
        
        W2VEMB_.append(z)

    W2VEMB_=np.array(W2VEMB_)
    print(W2VEMB_.shape)
    W2VEMB_=W2VEMB_.reshape(-1,300)
    print(W2VEMB_.shape)

    w2v_feature=pd.DataFrame(W2VEMB_)
     
    col=["W2V_F"+str(i+1) for i in range(0,300)]
    w2v_feature.columns=col
    w2v_feature=pd.concat([CLASS_,w2v_feature],axis=1)
    w2v_feature.index=PID_
    
    w2v_feature.to_csv(outFile)
    print("Word2Vec embedding finished@￥￥￥￥￥")
    print("it took %0.3f mins.\n"%((time.time()-T0)/60))

    return w2v_feature


 
