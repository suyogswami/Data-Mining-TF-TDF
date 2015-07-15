import sys
import os
import re
import math
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus_root = 'E:\\Data Mining Programming Assignment 1\\stateoftheunionaddresses'
fileapp=[]


def tokenize_str(stri):
    token=[]
    token1=[]
    j=0
    stopwordsset= sorted(stopwords.words('english'))
    stemmer = PorterStemmer()
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    token=tokenizer.tokenize(stri)
    token=[demo_token.lower() for demo_token in token]
    token=[demo_token for demo_token in token if not demo_token in stopwordsset]   
    for to2 in token:
        to3=stemmer.stem(to2)
        token1.insert(j,to3)
        j=j+1
    return(token1)

def tokenize_files(corpus_root): 
    tokens=[]
    stopwordsset= sorted(stopwords.words('english'))
    stemmer = PorterStemmer()
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')    
    for filename in os.listdir(corpus_root):
        token1=[]
        j=0
        i=0
        token=[]
        file = open(os.path.join(corpus_root, filename), "r")
        fileapp.append(filename)
        doc = file.read()
        token=tokenizer.tokenize(doc)
        token=[demo_token.lower() for demo_token in token]
        token=[demo_token for demo_token in token if not demo_token in stopwordsset]   
        for to2 in token:
            to3=stemmer.stem(to2)
            token1.insert(j,to3)
            j=j+1
        tokens.append(token1)
    return(tokens)

def tokenize_text(file_name): 
    stopwordsset= sorted(stopwords.words('english'))
    stemmer = PorterStemmer()
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')    
    token1=[]
    j=0
    for filename in os.listdir(corpus_root):
        if filename == file_name:
            file = open(os.path.join(corpus_root, filename), "r")
            doc=file.read()
            token=tokenizer.tokenize(doc)
            token=[demo_token.lower() for demo_token in token]
            token=[demo_token for demo_token in token if not demo_token in stopwordsset]   
            for to2 in token:
                to3=stemmer.stem(to2)
                token1.insert(j,to3)
                j=j+1
    return(token1)
    
def gettf(tokens):

    Matrix1=[]
    k=0
    token_length=len(tokens)
    Matrix = []
    for t2 in tokens: 
        token_count=tokens.count(t2)
        if token_count>0:
            tf=1.0 + math.log10(token_count)
        else:
            tf=0.0
        if any(e[0] == t2  for e in Matrix):
            pass 
        else:
            Matrix.append([t2,tf])
    return(Matrix)

def gettf_file(file_token):
    Matrix1=[]
    k=0
    file_token_length=len(file_token)
    Matrix = []
    for t2 in file_token: 
        token_count=file_token.count(t2)
        if token_count>0:
            tf=(1.0 + math.log10(token_count)) 
        else:
            tf=0.0
        if any(e[0] == t2  for e in Matrix):
           pass 
        else:
            Matrix.append([t2,tf])
                
    return(Matrix)


def gettf_files(tokens_file):
    Matrix1=[]
    Matrix = []
    for t1 in tokens_file:
        for t2 in t1: 
            token_count=t1.count(t2)
            if token_count>0:
                tf=1.0 + math.log10(token_count)
            else:
                tf=0.0
            if any(e[0] == t2  for e in Matrix):
                pass 
            else:
                Matrix.append([t2,tf])
        Matrix1.append(Matrix)
    return(Matrix1)

def getidf(token):
    token_all1=tokenize_files(corpus_root)
    number_of_token=0
    for tok in token_all1:
        if token in tok:
            number_of_token=number_of_token+1
    if number_of_token > 0:
        return math.log10(float(len(token_all1)) / number_of_token)
    else:
        return 0.0
    
def getidf_file(file_tf_matrix,all_files_token):

    idf_list=[]
    for f_t_matrix in file_tf_matrix:
        number_of_token=0
        for token1 in all_files_token:
            if f_t_matrix[0] in token1:
                number_of_token=number_of_token+1
        if number_of_token > 0:
            idf_list.append(math.log10(float(len(all_files_token)) / number_of_token))
        else:
            idf_list.append(0.0)
    return idf_list
        

def gettfidfvec(file_name):
    files_token=tokenize_files(corpus_root)
    i=0
    token_req=[]
    for file_token in files_token:
        if fileapp[i]==file_name:
            token_req=file_token
        i=i+1
    file_tf_matrix=gettf_file(token_req)  
    dictw={}
    f_t_matrix=[]
    token_idf=getidf_file(file_tf_matrix,files_token)
    
    n=0
    tfidf_sqr=0
    for f_t_matrix in file_tf_matrix:
        tfidf_sqr=tfidf_sqr + (token_idf[n] * f_t_matrix[1])**2
        n=n+1
        
    j=0 
    for f_t_matrix in file_tf_matrix:
        tfidf=token_idf[j] * f_t_matrix[1]
        tfidf_normalised= tfidf/math.sqrt(tfidf_sqr)
        if tfidf_normalised==0.0:
            pass
        else:
            dictw[f_t_matrix[0]]=tfidf_normalised
        j=j+1
    
    return(dictw)

def query(qstring): #return the document that has the highest similarity score with respect to 'qstring'.
    qstr_token=tokenize_str(qstring)
    queri_tf=gettf(qstr_token)
    files_token=tokenize_files(corpus_root)
    idf_que=getidf_file(queri_tf,files_token)
    #files_tf=gettf_files(files_token)
            
    sum_doc_tfidf=0
    documnt=0
    docs_ti_list=[]
    doc_ti_list=[]
    high_sim=[]
    i=0
    sum_query_tfidf=0
    query_ti=0
    quer_ti_list=[]
    q_tfidf=0
    cosine_sim=0
    l=0
    docu_tfidf=0
    
    for q_tf in queri_tf:
        sum_query_tfidf=sum_query_tfidf + (q_tf[1]*idf_que[i])**2
        i=i+1
        
    for doc_tokn in files_token:
        doc_tf=gettf_file(doc_tokn)
        j=0
        doc_idf=getidf_file(doc_tf,files_token)
        for d_tf in doc_tf:
            docu_tfidf=d_tf[1]*doc_idf[j]
            sum_doc_tfidf=sum_doc_tfidf + (docu_tfidf)**2
            k=0
            for q_tf in queri_tf:
                if q_tf[0] == d_tf[0]:
                    query_ti= q_tf[1]*idf_que[k]
                    cosine_sim=cosine_sim + (query_ti*docu_tfidf)
                k=k+1
            j=j+1
        if cosine_sim/math.sqrt(sum_query_tfidf)*math.sqrt(sum_doc_tfidf) > 0:
            cosine_sim_normalised=cosine_sim/math.sqrt(sum_query_tfidf)*math.sqrt(sum_doc_tfidf)
        else:
            cosine_sim_normalised = 0.0
        high_sim.append(cosine_sim_normalised)
    high_sim_file=high_sim.index(max(high_sim))
    return(fileapp[high_sim_file])
            

def querydocsim(queri,doc):
    qstr_token=tokenize_str(queri)
    query_tf=gettf(qstr_token)
    doc_token=tokenize_text(doc)
    document_tf= gettf_file(doc_token)
    files_token=tokenize_files(corpus_root)
    doc_idf=getidf_file(document_tf,files_token)
    que_idf=getidf_file(query_tf,files_token)
    
    doc_tfidf_list=[]
    i=0
    for d_tf in document_tf:
        doc_tfidf=d_tf[1]*doc_idf[i]
        doc_tfidf_list.append([d_tf[0],doc_tfidf])
        i=i+1
    
    que_tfidf_list=[]
    p=0
    for q_tf in query_tf:
        que_tfidf=q_tf[1]*que_idf[p]
        que_tfidf_list.append([q_tf[0],que_tfidf])
        p=p+1
    
    sum_d_tdidf=0
    for d_t_l in doc_tfidf_list:
        sum_d_tdidf=sum_d_tdidf + d_t_l[1]**2
    
    
    sum_q_tfidf=0
    for q_t_l in que_tfidf_list:
        sum_q_tfidf= sum_q_tfidf + q_t_l[1]**2    
    
    cosines=0
    cosine_normalised=0
    for d_l in doc_tfidf_list:
        for q_l in que_tfidf_list:
            if d_l[0]==q_l[0]:
                cosines = cosines + (d_l[1] * q_l[1])
    cosine_normalised = cosines / math.sqrt(sum_d_tdidf) * math.sqrt(sum_q_tfidf)  
    
    return(cosine_normalised)
                
    
def docdocsim(doc1,doc2):
    
    doc1_tokn=tokenize_text(doc1)
    doc2_tokn=tokenize_text(doc2)
    doc1_tf=gettf_file(doc1_tokn)
    doc2_tf=gettf_file(doc2_tokn)
    all_token=tokenize_files(corpus_root)
    doc1_idf=getidf_file(doc1_tf,all_token)
    doc2_idf=getidf_file(doc2_tf,all_token)
    
    x=0
    doc1_tfidf_list=[]
    for d1_tf in doc1_tf:
        doc1tfidf=doc1tfidf+(d1_tf[1]*doc1_idf[x])
        doc1_tfidf_list.append([d1_tf[0],doc1tfidf])
        x=x+1
        
    y=0
    doc2_tfidf_list=[]
    for d2_tf in doc2_tf:
        doc2tfidf=doc2tfidf+(d2_tf[1]*doc2_idf[y])
        doc2_tfidf_list.append([d2_tf[0],doc2tfidf])
        y=y+1
        
    sum_sqr_docti1=0
    for doct1 in doc1_tfidf_list:
        sum_sqr_docti1=sum_sqr_docti1+doct1[1]**2
        
    sum_sqr_docti2=0
    for doct2 in doc2_tfidf_list:
        sum_sqr_docti2=sum_sqr_docti2+doct2[1]**2
    
    cosine_d1d2=0
    cosine_norma=0
    for d1_til in doc1_tfidf_list:
        for d2_til in doc2_tfidf_list:
            if d2_til[0] == d1_til[0]:
                cosine_d1d2=cosine_d1d2+(d2_til[1] * d1_til[1])
                
    cosine_norma= cosine_d1d2/ math.sqrt(sum_sqr_docti1)*math.sqrt(sum_sqr_docti2)
    return(cosine_norma)
    
#print(getidf('health'))

#print(gettfidfvec('Barack ObamaJanuary 20, 2015.txt'))
 
#print(query('health insurance wall street'))

#print(docdocsim("Barack ObamaJanuary 20, 2015.txt", "Barack ObamaJanuary 28, 2014.txt"))

#print(querydocsim("health insurance wall street", "Barack ObamaJanuary 28, 2014.txt"))
