
# coding: utf-8

# # Rule Based Association Pattern mining to find the sumarized the terms and condition
# Team members:
# Nikhil Gola(MT18129) Ridha Juneja(MT18009) Saru Brar(MT18014) Yogesh Pandey(MT18140)
# *****************************************************************************************
# 
# Importing all the necessary libraries required for the Rule based Association Pattern mining

# In[1]:


import numpy as np  
import matplotlib.pyplot as plt  
import pickle
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np


# In[3]:


import os
import re
import copy
import time


# Setting the min_support and min_confidence value for the Confidence table

# In[4]:


min_support=0.4
min_confidence=0.70


# Loading the pickle word dictionary pickle file after preprocessing of the data (pickle file will act as a cache to our System)

# In[5]:


def loadWordDictionaryPickle():
    f=open('Pickled Data/word_Dictionary.pickle','rb')
    vocab=pickle.load(f)
    f.close()
    return vocab


# Loading the word frequncy pickle file from preprocessed data

# In[6]:


def loadWordFrequencyPickle():
    f1=open('Pickled Data/word_Frequency.pickle','rb')
    DocVocab=pickle.load(f1)
    #print(DocVocab['Facebook'])
    f1.close()
    return DocVocab


# Loading the unique word pickle from the preprocessed data

# In[7]:


def loadUniqueWordPickle(): 
    f1=open('Pickled Data/unique_Wordset.pickle','rb')
    unique_word=pickle.load(f1)
    unique_word=list(unique_word)
    f1.close()
    return unique_word


# Calculating word frequency in the training set(corpus) and calculating the probability of word for occuring in corpus and the tfidf values for every word in the corpus

# In[8]:


def uniqueWordFrequency(DocVocab):
    Word_Set_frequency={}
    wordSetSupport = {}
    tfidf_dict={}
    maxDoc=len(DocVocab.keys())
    print(maxDoc)
    for key in DocVocab.keys():
        for word in DocVocab[key]:
            if Word_Set_frequency.get(word) ==None:
                Word_Set_frequency[word]=DocVocab[key][word]
                if DocVocab[key][word]==0:
                    wordSetSupport[word]=0.0
                else:
                    wordSetSupport[word]=1.0
            else:
                if DocVocab[key][word]==0:
                    wordSetSupport[word]+=0.0
                else:
                    wordSetSupport[word]+=1.0
                Word_Set_frequency[word]+=DocVocab[key][word]
    #print(wordSetSupport['combine'])
    total1_sum=np.sum([Word_Set_frequency[key] for key in Word_Set_frequency.keys()])
    for key in wordSetSupport.keys():
            wordSetSupport[key]=wordSetSupport[key]/maxDoc
            tfidf_dict[key]=wordSetSupport[key]*(Word_Set_frequency[key]/(1.0*total1_sum))
    return Word_Set_frequency,wordSetSupport,tfidf_dict
        


# Filtering the words from the wordset based upon their frequency which above the mean of frequency

# In[9]:




#print([Word_Set_frequency[key]  for key in Word_Set_frequency.keys()])

#print(mean1)
#median1=np.median([Word_Set_frequency[key]  for key in Word_Set_frequency.keys()])
#print(median1)
#threshold=np.array([Word_Set_frequency[key]  for key in Word_Set_frequency.keys()])
#threshold1=np.reshape(len(threshold),1)
#kmeans = KMeans(n_clusters=4, random_state=0).fit(threshold1)
#print(kmeans)
def filterWordSetFrequencySupport(mean1,Word_Set_frequency,wordSetSupport):
    req_Dic={}
    req_Dic_Sup={}
    for key in Word_Set_frequency.keys():
        if Word_Set_frequency[key]>mean1:
            req_Dic[key]=Word_Set_frequency[key]
    for key in req_Dic.keys():
        req_Dic_Sup[key]=wordSetSupport[key]
    return req_Dic,req_Dic_Sup


# In[10]:


'''fp = open('TFIDF_values.pickle','rb')
tfidf_values={}
tfidf = pickle.load(fp)
#print(tfidf['Google'])
for key in tfidf.keys():
    for wordkey in tfidf[key].keys():
        if tfidf_values.get(wordkey)==None:
            tfidf_values[wordkey]=tfidf[key][wordkey]
        else:
            tfidf_values[wordkey]+=tfidf[key][wordkey]
for key in tfidf_values.keys():
    tfidf_values[key] = tfidf_values[key]/len(tfidf.keys())
#print(tfidf_values[''])
'''


# In[11]:


'''def returnItemsWithMinSupport(vocab):
        maxDoc=len(vocab.keys())
        Support_Score={}
        for key in vocab.keys():
            for words in vocab[key]:
                if Support_Score.get(words) is None:
                    Support_Score[words]=1
                else:
                    Support_Score[words]+=1
        for key in Support_Score.keys():
            Support_Score[key]=Support_Score[key]/maxDoc
        return Support_Score
'''


# In[12]:


def getData():
    words=[]
    with open("data1.txt","r") as f:
        x  = f.readlines()
        for word in x:
            words.append(word.replace("\n",""))
            
    return words


# In[13]:


def FilterDictionary(A,Support_Score,unique_word):
    filteredDict=[]
    SScore={}
    for word in A:
        if word in unique_word:
            filteredDict.append(word)
            SScore[word]=Support_Score[word]
    return filteredDict,SScore


# In[14]:


def LemmatizeWord(A):
    WN=WordNetLemmatizer()
    i=0;
    maxLen=len(A)
    while i<maxLen:
        A[i]=WN.lemmatize(A[i])
        i+=1
    return A
        


# In[15]:


def ListToDic(vocab):
    my_dict={}
    for doc in vocab.keys():
        my_dict = {k: 0 for k in vocab[doc] }
        vocab[doc]=my_dict
    
    return vocab


# In[16]:


def confidence(A,B,vocab):
    ScoreMatrix=np.zeros((len(A),len(B)))
    #print(len(B))
    #print(len(A))
    for index1,word1 in enumerate(A):# A is list of words
        for index2,word2 in enumerate(B):
            common=0
            totA=0
            for doc in vocab.keys():
                if  vocab[doc].get(word1) is not None :
                    totA+=1
                    if vocab[doc].get(word2) is not None:
                        common+=1
                    
            if totA==0:
                print(word1,word2)
            ScoreMatrix[index1][index2]=common/totA
    return ScoreMatrix


# In[17]:



#print(len(A))
#print(requiredDicti['software'])
#print('software' in B )
#print(len(B))
#print(B)
#print(Ascore)


# In[18]:


def SelectAssociationRules(confidenceMatrix,min_support,min_confidence,A,B,Support_Score):
    row,col=confidenceMatrix.shape
    #print(row,col)
    required={}
    for i in range(0,row):
        for j in range(0,col):
            word=A[i]
            score=Support_Score[word]
            word2=B[j]
            if confidenceMatrix[i][j]>=min_confidence and score>min_support:
                if required.get(word) ==None and (word!='' and word2!='') :
                    required[word]=[word2]
                else:
                    if (word!='' and word2!=''):
                        required[word].append(word2)
                #print(word,'-->',word2)
                #print("\n")
    return required


# In[19]:


'''def MeanOFMean(confidenceMatrix):
    mean1=[]
    row,col=confidenceMatrix.shape
    for i in range(0,row):
            mean1.append(np.median(confidenceMatrix[i]))
#             print(mean1)
    return np.mean(mean1)
MeanOFMean(confidenceMatrix)
'''


# In[20]:


def Find(string): 
    # findall() has been used  
    # with valid conditions for urls in string 
    url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+] |[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string) 
    return url 


# In[56]:


def printSummary(rule_Dict,outputfile):
    list_of_lines_to_print = []
    #print(rule_Dict.keys())
    
    #print(rule_Dict['personal_information']['DuckDuckGo does not collect or share personal information'])
    
    #rule_Dict[key][key2] for key2 in rule_Dict[key]
    with open(outputfile,"w") as file1:
        for key in rule_Dict.keys():
            testList=rule_Dict[key]
            #print([ele[0] for ele in testList])
            mean8 = np.mean([ele[0][1] for ele in testList])
            #print(mean8)
            for rows in testList:
                #print(rows[0])
                if rows[0][1]>=mean8:
                    #rows[0][0]=rows[0][0].decode('utf-8','ignore')
                    if rows[0][0] not in list_of_lines_to_print:
                        file1.write(rows[0][0]+'\n\n')
                        list_of_lines_to_print.append(rows[0][0])
                    
                    
            


# In[57]:



def doSummarization(filename,filepath,rules,tfidf_dict):
    outputfile = filepath+'summary/'+filename[:-4]+'associationSummary'+'.txt'
    filename = filepath+filename
    linesData =[]
    rule_Dict={}
    with open(filename,encoding="utf8") as file:
        data = file.readlines()
        #print(data)
        
        for line2 in data:
            linedata=line2.split(".")
            for line in linedata:
                s = Find(line)
                #print("url:", s)
                if len(s)<=0:
                    
                    line = line.replace("\n","")
                    line = line.replace("-","")
                    line1=line.lower()
                    #print("line:",line)
                    tokens=nltk.word_tokenize(line1)
                    if len(tokens)>1:
                        tokens=nltk.word_tokenize(line)
                        #print(tokens)
                        flag=0
                        weight=0
                        line_dict=[]
                        for token in tokens:
                            if rules.get(token)!=None:
                                for word2 in rules[token]:
                                    if word2 in tokens:
                                        flag=1
                                        #print("Printing this line beacuse of ",token,"-->",word2)
                                        #print('--> ',line,".\n")
                                        linesData.append(line)
                                        for token1 in tokens:
                                            if tfidf_dict.get(token1) is not None:
                                                weight=weight+tfidf_dict[token1]
                                        line_dict.append([line,weight])
                                        break
                            if flag==1:
                                break
                        
                        if flag==1:
                                w1=token.lower()
                                w2=word2.lower()
                                str1=w1+'_'+w2
                                if rule_Dict.get(str1) is None:
                                    rule_Dict[str1]=[line_dict]
                                else:
                                    rule_Dict[str1].append(line_dict)

    #print(len(linesData))
   
    printSummary(rule_Dict,outputfile)


# In[58]:


def main():
    vocab = loadWordDictionaryPickle()
    Support_Score={}
    DocVocab = loadWordFrequencyPickle()
    Word_Set_frequency,wordSetSupport,tfidf_dict=uniqueWordFrequency(DocVocab)
    mean1=np.mean([Word_Set_frequency[key]  for key in Word_Set_frequency.keys()])
    req_Dic,req_Dic_Sup = filterWordSetFrequencySupport(mean1,Word_Set_frequency,wordSetSupport)
    unique_word = loadUniqueWordPickle()
    vocab=ListToDic(vocab)
    A=getData()
    requiredDicti ={}
    A,Ascore=FilterDictionary(A,wordSetSupport,unique_word)
    A=LemmatizeWord(A)
    for key in req_Dic.keys():
        if key not in A:
            requiredDicti[key]=req_Dic[key]
    B=[]
    for item,value in requiredDicti.items():
        B.append(item)
    confidenceMatrix=confidence(A,B,vocab)
    print(confidenceMatrix)
    rules=SelectAssociationRules(confidenceMatrix,min_support,min_confidence,A,B,wordSetSupport)
    test_path = "Datasets/testdata/"
    test_files = os.listdir(test_path)  
    print(test_files)
    #doSummarization("Datasets/testdata/Github.txt")
    for file in test_files:
        if os.path.isdir(test_path+file):
            pass
        else:
            print("doing for ",file)
            doSummarization(file,test_path,rules,tfidf_dict)


# In[59]:

#Similarity********************************************************************
def similarity(file1,file2,name1,name2):
    dict1 = getTokensDictFromFile(file1)
    dict2 = getTokensDictFromFile(file2)
    sim = getSimilarity(dict1,dict2)
    acc = sim*100
    print("Accuracy for ",name1," and ",name2," is : ",acc,"%")


def getTokensDictFromFile(file):
    data = open(file).read()
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    tokens = word_tokenize(data.lower().translate(remove_punctuation_map))
    words = [word.lower() for word in tokens]
    stemmer = nltk.stem.porter.PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words]
    
    stopword = set(stopwords.words('english'))
    filtered_words = [word for word in stemmed_words if not word in stopword]
    
    mydict = nltk.defaultdict(int)
    for word in filtered_words:
        mydict[word]+=1
    return mydict
    
    


# In[27]:


def cosine_similarity(a,b):
    dot_product = np.dot(a,b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    return dot_product/(norma*normb)


# In[28]:


def getSimilarity(dict1,dict2):
    words =[]
    for key in dict1.keys():
        words.append(key)
    for key in dict2.keys():
        words.append(key)
    n = len(words)
    vector1 = np.zeros(n,dtype=np.int)
    vector2 = np.zeros(n,dtype=np.int)
    i=0
    for (key) in words:
        vector1[i] = dict1.get(key,0)
        vector2[i] = dict2.get(key,0)
        i=i+1
    sim = cosine_similarity(vector1,vector2)
    return sim


# In[60]:
#END Similarity********************************************************************

main()

sum_path = "Datasets/testdata/summary/"
grnd_path ="Datasets/GroundTruth/"
sum_files = os.listdir(sum_path)
grnd_files = os.listdir(grnd_path)

#print(sum_files)
#print(grnd_path)
#doSummarization("Datasets/testdata/Github.txt")
for i in range(0,len(sum_files)):
    if os.path.isdir(sum_path+sum_files[i]):
        pass
    else:
        similarity(sum_path+sum_files[i],grnd_path+grnd_files[i],sum_files[i],grnd_files[i])
        


# In[26]:

