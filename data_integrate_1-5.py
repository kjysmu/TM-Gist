# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 11:12:09 2016

@author: YM
"""


import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd
import twokenize



def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    blank_word=True
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
        elif blank_word==True:
            x.append(word_idx_map['blank_word'])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x



def make_idx_data_cv(revs, word_idx_map, cv, max_l=56, k=300, filter_h=8):
    """
    Transforms sentences into a 2-d matrix.
    """
    X_train,Y_train,X_test,Y_test = [],[],[], []

    for rev in revs:
        if rev["y"]==3:
            rev["y"]=2        
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)
        
        if rev["split"] == cv:
            X_test.append(sent)
            Y_test.append(rev["y"])
        else:
            X_train.append(sent)
            Y_train.append(rev["y"])
            
            
    X_train = np.array([X_train], dtype="int")
    Y_train = np.array(Y_train, dtype="int")
    X_test = np.array([X_test], dtype="int")    
    Y_test = np.array(Y_test, dtype="int")  
    
    X_train=X_train.swapaxes(1,0)     
    X_test=X_test.swapaxes(1,0)       
    
    return X_train,Y_train,X_test,Y_test

def make_idx_data_TT(revs, word_idx_map, max_l=72, k=300, filter_h=8):
    """
    Transforms sentences into a 2-d matrix.
    """
    X_train,Y_train,X_test,Y_test = [],[],[], []
    X_pretrain,Y_pretrain=[],[]
    for rev in revs:
        #if rev["y"]%10==2: # neutral removing
        #    continue
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)

        if rev["y"]>=20:
            X_pretrain.append(sent)
            Y_pretrain.append(rev["y"]-20)
        elif rev["y"] >= 10:
            X_test.append(sent)
            Y_test.append(rev["y"]-10)
        else:
            X_train.append(sent)
            Y_train.append(rev["y"])
            
            
    X_train = np.array([X_train], dtype="int")
    Y_train = np.array(Y_train, dtype="int")
    X_test = np.array([X_test], dtype="int")    
    Y_test = np.array(Y_test, dtype="int")  
    X_pretrain = np.array([X_pretrain], dtype="int")    
    Y_pretrain = np.array(Y_pretrain, dtype="int")  

    X_train=X_train.swapaxes(1,0)     
    X_test=X_test.swapaxes(1,0) 
    X_pretrain=X_pretrain.swapaxes(1,0)  
    
    return X_train,Y_train,X_test,Y_test,X_pretrain,Y_pretrain





def build_data_cv(data_folder, clean_string=True,vocab=defaultdict(float),plus=0):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    i=0
    #vocab = defaultdict(float)
    with open(data_folder, "rb") as f:
        for line in f:
            i+=1
            line=line.strip()
            rev = []
            line=line.split('\t')
            if len(line)>4 or len(line)<3:
                print data_folder
                print line,'\t',i
                break
            if line[2]=='positive':
                y=0
            elif line[2]=='negative':
                y=1
            elif line[2]=='neutral':
                continue # for task B
                y=2
            else:
                #print 'class error', line
                continue
            
            
            rev.append(line[3].strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev)).strip()
            else:
                orig_rev = " ".join(rev).lower()
            if orig_rev =="not available":
                continue    
            if orig_rev!="":
                words = set(orig_rev.split())
                for word in words:
                    vocab[word] += 1
                datum  = {"y":y+plus, 
                          "text": orig_rev,                             
                          "num_words": len(orig_rev.split())}         
                         
                
                # upsampling                
                #if y==1 and plus==0:
                #    revs.append(datum)
                #    revs.append(datum)
                
                    
                revs.append(datum)
                if y==1 and plus==0:
                    revs.append(datum)

    return revs, vocab
    
def get_W(word_vecs, k=100):
    """
    depending on word2vec list
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map
    
    
def get_W2(word_vecs,vocab=defaultdict(float), min_df=10,k=100):
    """
    depending on vocab list
    """
    word_idx_map = dict()
    W = []       
    W.append([0]*k)# 00000000
    W.append(np.random.uniform(-0.25,0.25,k)) # infrequency word
    i = 2
    for word in vocab:
        if word in word_vecs:
            W.append(word_vecs[word])
            word_idx_map[word] = i     
            i += 1   
        elif vocab[word] >= min_df:
            #print word
            W.append(np.random.uniform(-0.25,0.25,k))
            word_idx_map[word] = i            
            i += 1    
            # do addong like verb noun etc
    W = np.array(W,dtype='float32')
    return W, word_idx_map

def get_W2forsecond(word_vecs,word_idx_map=[], min_df=10,k=100):
    """
    to use same word idx
    """
    
    W = []       
    W.append([0]*k)# 00000000
    W.append(np.random.uniform(-0.25,0.25,k)) # infrequency word
    i = 2
    for word in word_idx_map:
        if word in word_vecs:
            W.append(word_vecs[word])             
        else:
            W.append(np.random.uniform(-0.25,0.25,k))
        i += 1     
    return np.array(W,dtype='float32')



def get_W3(W,word_idx_map):    
    """
    for sentiScore attaching
    """
    f=open("data/sentiScore.txt")
    senti_dic= defaultdict(float)
    while True:
        line=f.readline()
        if not line: break
        line=line.split('\t')
        senti_dic[line[0]]=line[1][:-1]
    f.close()
    W=W.tolist()    
    i=0
    W[0].append('0.0')
    W[1].append('0.0')    
    for word in word_idx_map:
        if word in senti_dic:
            W[word_idx_map[word]].append(senti_dic[word])      
        else:
            W[word_idx_map[word]].append('0.0') 
        i+=1
    return np.array(W,dtype='float32')


def load_bin_vec(fname, vocab,binary=True):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    if binary == True:
        with open(fname, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size
            for line in xrange(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)   
                if word in vocab:
                   word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
                else:
                    f.read(binary_len)
    else:
        with open(fname, "r") as f:
            while True:
                line=f.readline()
                if not line: break
                line=line.split(' ')
                if line[0] in vocab:
                    word_vecs[line[0]]=np.array(line[1:],dtype='float32') 
            #header = f.readline()
            #vocab_size, layer1_size = 1193515,100#map(int, header.split())
        
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=10, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = twokenize.tokenize(string.lower())
    for i in range(len(string)):
        if string[i][0] == '@':
            string[i] = ""
        elif string[i][0:4] =="http":
            string[i]=""
    string = " ".join(string)


    """
    reducing repeated char
    """
    a=string[0]    
    b=string[1]
    newStr=a+b
    
    for i in range(len(string)-2):
        c=string[i+2]
        if(a==b and b==c):
            pass
        else:
            newStr=newStr+c
        a=b
        b=c
    string = newStr
    
    string = string.replace("`","\'")
    string = string.replace("\u002c",",")
    string = string.replace("\u2019","\'")
    string = string.replace("\\\"\"","\"")
    
    '''
    string = re.sub(r"[^A-Za-z0-9(),!?]", " ", string)     
    
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!+", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)
    '''
    
    string = twokenize.tokenize(string)
    return " ".join(string)
    
    
def unescape(text):
    text = text.replace('’', '\'')
    text = text.replace('“', '"')
    text = text.replace('…', '...')
    return text


def normalize_special(text):
    text = re.sub(r'\S{,4}://\S+', ' http://someurl ', text)
    text = re.sub(r'[@＠][a-zA-Z0-9_]+:?', ' @someuser ', text)
    return text


def normalize_elongations(text):
    text = re.sub(r'([a-zA-Z])\{2,}', r'\1\1', text)
    text = re.sub(r'\.{4,}', r'...', text)
    return text

if __name__=="__main__":    
    
    w2v_file = "F:/word2vec/word2vec_google/GoogleNews-vectors-negative300.bin"
    #w2v_file = "data/word2vec.bin"    
    #w2v_file = "data/word2vec_pret_small3.txt"
    globe_file = "D:/glove.twitter.27B/glove.twitter.27B.100d.txt"
    print "loading data...",  
    
    # For task A
    data_sub_path = "F:/semEval_code/data/GOLD/"      
    training_file1 = data_sub_path + "Subtasks_BD/twitter-2015train-BD.txt"
    revs1, vocab = build_data_cv(training_file1,  clean_string=True,plus=0)
    
    training_file2 = data_sub_path + "Subtasks_BD/twitter-2016train-BD.txt"
    revs2, vocab = build_data_cv(training_file2,  clean_string=True,vocab=vocab,plus=0)
    '''
    training_file3 = data_sub_path + "Subtasks_BD/twitter-2016train-A.txt"
    revs3, vocab = build_data_cv(training_file3,  clean_string=True,vocab=vocab,plus=0)
    '''
    test_file=data_sub_path + "Subtasks_BD/twitter-2016test-BD.txt"
    revs_test, vocab = build_data_cv(test_file, clean_string=True,vocab=vocab,plus=10)
    
    dev_file1=data_sub_path + "Subtasks_BD/twitter-2016devtest-BD_modified.txt"
    revs_dev1, vocab = build_data_cv(dev_file1, clean_string=True,vocab=vocab,plus=20)
    '''
    dev_file2=data_sub_path + "Subtasks_BD/twitter-2016dev-A.txt"
    revs_dev2, vocab = build_data_cv(dev_file2, clean_string=True,vocab=vocab,plus=20)
    '''
    #pretrain_file="C:/Lasagne/code/distance sup_data/cleaned.txt"
    #revs3, vocab = build_data_cv(pretrain_file,  clean_string=True,vocab=vocab,plus=20)

    vocab['blank_word']=100 # enter the blank_word to change unfrequency word
    revs=revs1+revs2+revs_test+revs_dev1
    
    print "counting..."
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    
    
    
    print "loading word2vec vectors..."
    w2v = load_bin_vec(w2v_file, vocab,binary=True)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    #add_unknown_words(w2v, vocab,min_df=5,k=100)
    W, word_idx_map = get_W2(w2v,vocab, min_df=5,k=300)
    #W = get_W3(W,word_idx_map) # add 1 - dimension , now w2v contain it from pretraining
    '''
    print "loading word2vec vectors...",
    w2v2 = load_bin_vec(globe_file, vocab,binary=False)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    #add_unknown_words(w2v, vocab,min_df=5,k=100)
    W2 = get_W2forsecond(w2v2,word_idx_map, min_df=1,k=100)    
    W2 = get_W3(W2,word_idx_map)
    '''
    #cPickle.dump([revs, W, W2, word_idx_map, vocab], open("C:/Lasagne/code/data/mr_4.p", "wb"))
    
    
    k,kernel_size = 300, 5  
    cPickle.dump([revs,word_idx_map,W],open("C:/Lasagne/code/data/mp.data", "wb"))
    '''
    X_train,Y_train,X_test,Y_test,X_dev,Y_dev = \
                    make_idx_data_TT(revs, word_idx_map, max_l=max_l, k=k, filter_h=kernel_size)
    '''
    
    
    #cPickle.dump([W,W2,word_idx_map],open("C:/Lasagne/code/data/Wordvec&map3.p", "wb"))
    #cPickle.dump([ X_train,Y_train,X_test,Y_test], open("C:/Lasagne/code/data/mr_integ_TT3.p", "wb"))
    #cPickle.dump([ X_pretrain,Y_pretrain], open("C:/Lasagne/code/data/mr_integ_pret3.p", "wb"))
    print "dataset created!"
    

'''
for rev in revs:
    text = rev["text"]
    if len(twokenize.tokenize(text))>80:
        print text
'''
i=0
for label in Y_train:
    if label ==1:
        i +=1    
print i