
# coding: utf-8

# In[19]:


import os
import math
import pandas as pd
import copy
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
from nltk import pos_tag
from collections import Counter
import numpy as np


# In[3]:


import nltk
nltk.download('stopwords')


# In[4]:


nltk.download('averaged_perceptron_tagger')


# In[5]:


nltk.download('wordnet')


# In[20]:


data_files=[]
path = os.path.abspath(os.path.dirname('/home/ridha/TextSummarizer/Datasets'))                   #Path of data set folder
data_set_path = os.path.join(path, "Datasets/Webapps Data")
def Filepath():
        data_files = copy.deepcopy(os.listdir(data_set_path))
        return data_files


# In[21]:


def fileRead(data_file):                                                 #Method to read all the dataset files 
            actual_path = data_set_path +"/"+ data_file
            new_file__content =  open(actual_path,"rb")
            file_content=' '
            for nffile in new_file__content:
                nffile=nffile.decode('utf-8', 'ignore')
                file_content+=nffile
            file__content=copy.deepcopy(file_content)
            sentence_set =[]
            sentence_set = nltk.sent_tokenize(file_content)
            return sentence_set


# In[22]:


def readingFiles():
    dictFile={}
    for file in data_files:

        dictFile[file[:-4]]=fileRead(file)
    return dictFile


# In[23]:


data_files=Filepath()
print(data_files)


# In[24]:


dic=readingFiles()
#print(dic)
column=[key for key in dic.keys()]
print(len(dic[column[1]]))


# In[25]:


dict1={'1':'b','2':'f'}
if dict1.get('2') is not  None:
    print('yes')


# In[26]:


dataframe = pd.DataFrame(list(dic.items()),columns=['softwareName','termsndCondition'])
print(dataframe.head(10))


# In[27]:


vocabDic={}
stop_words=set(stopwords.words("english")) 


# In[28]:


for license in dataframe['termsndCondition']:
    for sentence  in license:
        for word in nltk.word_tokenize(sentence):
            word=word.lower()
            if word not in stop_words and len(word)>2 and word.isalpha() and word!= ' ':                         
                if vocabDic.get(word) is not  None:
                    vocabDic[word]+=1
                else:
                    vocabDic[word]=0


# In[29]:


print(len(vocabDic))
freqList=[vocabDic[key] for key in vocabDic.keys()]
print(np.median(freqList))
freqmean=np.mean(freqList)
reqDic={}
for key,value in enumerate(vocabDic):
    if vocabDic[value] >= freqmean:
        reqDic[value]=vocabDic[value]
print(len(reqDic))
print(reqDic)


# In[66]:


proccesedSentenceDic={}
for  index,license in dataframe.iterrows():
   # print(len(license['termsndCondition']))
    z=0
    sentenceList=[]
    for index,sentence  in enumerate(license['termsndCondition']):
            z+=1
            for word in nltk.word_tokenize(sentence):
                word=word.lower()                        
                if reqDic.get(word) is not None:
                    sentenceList.append(sentence)
                    break
    #print(z)
    proccesedSentenceDic[license['softwareName']]=sentenceList


# In[35]:


#for key,value in enumerate( proccesedSentenceDic):
#    print(value,len(proccesedSentenceDic[value]))


# In[36]:


#print(len(proccesedSentenceDic['Ola']))


# In[67]:


uniqueWordData={}
for  index,license in dataframe.iterrows():
    softwareDocs =license['termsndCondition']
    vocabdic={}
    for sentence in softwareDocs:
        for sentWord in nltk.word_tokenize(sentence):
             if vocabdic.get(sentWord) is None and sentWord not in stop_words and len(sentWord)>2 and sentWord.isalpha() and sentWord!= ' '  :
                    vocabdic[sentWord]=1
    uniqueWordData[license['softwareName']]=vocabdic


# In[48]:


#print(uniqueWordData)


# In[70]:


idfScore={}
maxDoc=len(uniqueWordData.keys())
for key in uniqueWordData.keys():
    #print(key)
    for words in uniqueWordData[key].keys():
       # print(words)
        if reqDic.get(words) is not None :
            if idfScore.get(words) is None:
                idfScore[words]=1
            else:
                idfScore[words]+=1


# In[71]:


for key in idfScore.keys():
    idfScore[key]=(idfScore[key]+1)/maxDoc


# In[83]:


print(len(idfScore))


# In[73]:


idfValue=[]
idfValue=[idfScore[key] for key in idfScore.keys() ]


# In[80]:


freqmedian=np.median(idfValue)


# In[81]:


freqmean=np.mean(idfValue)


# In[110]:


reqDicIdf={}
for key in idfScore.keys():
   if idfScore[key] >= freqmedian:
       reqDicIdf[key]=idfScore[key]
print(len(reqDicIdf))


# In[111]:


reqDicIdf={}
for key in idfScore.keys():
   if idfScore[key] > freqmean:
       reqDicIdf[key]=idfScore[key]
print(len(reqDicIdf))


# In[112]:


len(proccesedSentenceDic['Youtube'])


# In[113]:


newProccesed={}
for processedTerms in  proccesedSentenceDic.keys():
   # print(processedTerms)
    sentenceList=[]
    for sentence  in proccesedSentenceDic[processedTerms]:
            for word in nltk.word_tokenize(sentence):
                word=word.lower()                        
                if reqDicIdf.get(word) is not None:
                    sentenceList.append(sentence)
                    break
    newProccesed[processedTerms]=sentenceList


# In[114]:


print(len(newProccesed['Youtube']))


# In[115]:


reqDicScore={}
for key in reqDicIdf.keys():
   reqDicScore[key]=reqDicIdf[key]*reqDic[key]


# In[116]:


print(reqDicScore)


# In[117]:


value=[reqDicScore[key] for key in reqDicScore]


# In[119]:


freqmean=np.mean(value)


# In[121]:


reqDicMScore={}
for key in reqDicScore.keys():
   if reqDicScore[key] > freqmean:
       reqDicMScore[key]=reqDicScore[key]
print(len(reqDicMScore))


# In[122]:


secondProcessed={}
for processedTerms in  newProccesed.keys():
   # print(processedTerms)
    sentenceList=[]
    for sentence  in newProccesed[processedTerms]:
            for word in nltk.word_tokenize(sentence):
                word=word.lower()                        
                if reqDicMScore.get(word) is not None:
                    sentenceList.append(sentence)
                    break
    secondProcessed[processedTerms]=sentenceList


# In[123]:


len(secondProcessed['Ola'])


# In[124]:


len(secondProcessed['Youtube'])


# In[125]:


secondProcessed['Youtube']

