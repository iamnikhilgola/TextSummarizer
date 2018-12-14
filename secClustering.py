import os
import pickle
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

path = os.path.abspath(os.path.dirname(__file__))  
pickle_path = os.path.join(path, "Pickled Data")
summary_path = os.path.join(path,  "summary")
ground_path = os.path.join(path, "groundTruth")

with open(pickle_path+"/test_dictionary.pickle","rb") as pickle_in:
    test_dictionary = pickle.load(pickle_in)
    
with open(pickle_path+"/K_MeansCluster.pickle","rb") as pickle_in:
    kmCluster = pickle.load(pickle_in)
    
with open(pickle_path+"/sentence_Dictionary.pickle","rb") as pickle_in:
    file_sentence_dictionary = pickle.load(pickle_in)
    
with open(pickle_path+"/word_Dictionary.pickle","rb") as pickle_in:
    file_word_dictionary = pickle.load(pickle_in)

with open(pickle_path+"/word_Frequency.pickle","rb") as pickle_in:
    file_word_frequency = pickle.load(pickle_in)
    
with open(pickle_path+"/file_content.pickle","rb") as pickle_in:
    file_content = pickle.load(pickle_in)

with open(pickle_path+"/fileNames.pickle","rb") as pickle_in:
    fileNames = pickle.load(pickle_in)

def computeLabel(word,score):                   #To compute label for a word
    for value in kmCluster:
        if word in kmCluster[value]:
            return value
 
for cluster in kmCluster:
    if 'privacy' in kmCluster[cluster]:
        priC= cluster
    if 'agreement' in kmCluster[cluster]:
        accC = cluster

print(priC,accC)
        
imp_set=[]
data = open("data1.txt","rb")                   #reading self made vocubalary
for nffile in data:
    nffile=nffile.decode('utf-8', 'ignore')
    imp_set.append(nffile.strip("\n\r"))
test_set=[]
imp_set=list(set(imp_set))                      #making set of self made vocubalory

kmCluster[len(kmCluster)]=imp_set               #making self made words as 9th cluster
        
clusterScore={}
for value in kmCluster:                         #giving each cluster a score
     clusterScore[value]=1/len(kmCluster[value])   
     
uniqueWordC=[]                                  #taking words of the important clusters i.e 4th and 6th
for lab in kmCluster:
    #if lab==2.0 or lab ==6.0:
    for words in kmCluster[lab]:
        uniqueWordC.append(words)
            
#for files in file_sentence_dictionary:          #Reading each file from the corpus
for files in test_dictionary:    
    dictFreqWC={}                                   #contains cluster value for each sentence
    dictLabels ={}                                  #contains nuetralized freq of imp words for each sentence 
    for Datas in test_dictionary[files]:   #Reading each sentence from the dictionary
        Data=word_tokenize(Datas)                   #tokenizing the sentence
        newDict={}                                  #Datastructure that notes the occurance of cluster words
        freqDict={}                                 #Datastructure that notes ocuurance of important words in a sentence
        for uwc in uniqueWordC:                     #initialize each word count as 0
            freqDict[uwc]=0
        for word in Data:
            value=computeLabel(word,clusterScore)   #gives cluster value of word,return 'none' when not in cluster
            if word in uniqueWordC:                 #Assigning frequency of particular word
                freqDict[word]+=1
            else:
                freqDict[word]=1
                
            if value in newDict:                    #frequency of each word from the cluster
                newDict[value]+=1
            else:                                   
                newDict[value]=1
        leng=len(Datas)                             #lenght of the Sentence
        for uwc in uniqueWordC:                     #nuetralizing the frequency count  by dividing each sentence by lenght
            freqDict[uwc]/=leng
        maxkey=0
        maxvalue=0
        for key in newDict:                         #code to select maximum label value of a sentence
            if(key!=None and newDict[key]>maxvalue):
                if(key==priC):
                    maxkey+=newDict[key]*40*clusterScore[key]
                elif(key==accC):
                    maxkey+=newDict[key]*40*clusterScore[key]
                else:
                    maxkey+=newDict[key]*1*clusterScore[key]
    
        maxkey=int(maxkey)
        if(maxkey>10):
            maxkey=maxkey/10
        dictLabels[Datas]=int(maxkey)                            #currently storing clustering value for each sentence
        if dictLabels[Datas]==priC or dictLabels[Datas]==accC:
            dictFreqWC[Datas]=freqDict                              #currently storing nuetralized frequency of each sentence
               
        
        usefull_labels={}  #assign new labels which correspond to our important marked sentences
        useless_words =['please','following','under these terms','payment terms','applicable laws','these terms','under', 'these','report']
        privacy=[]       # would store sentences related to privacy
        agreement=[]     # would store sentences related to agreement
        
        for nm in dictLabels:    #Taking each sentence from our labelled sentence
            flag=0
            if dictLabels[nm]==priC :
                w=word_tokenize(nm)
                for wo in w:
                    if wo in useless_words:
                        flag=1
                        if nm in dictFreqWC:
                            del dictFreqWC[nm]
                        break
                if flag==0:
                    #if nm not in usefull_labels:
                    usefull_labels[nm]=1
                    privacy.append(nm)
            flag=0          
            if dictLabels[nm]==accC :       
                w=word_tokenize(nm)
                for wo in w:
                    if wo in useless_words:
                        flag=1
                        if nm in dictFreqWC:
                            del dictFreqWC[nm]
                        break
                if flag==0:
                    #if nm not in usefull_labels:
                    usefull_labels[nm]=2
                    privacy.append(nm)           
            
        
        outputFile = open(summary_path+"/"+files+".txt","w")
        for values in privacy:
            outputFile.write(values+"\n\n\n")
        outputFile.close()

# In[59]:

#Similarity********************************************************************
def similarity(file1,file2,name1,name2):
    dict1 = getTokensDictFromFile(file1)
    dict2 = getTokensDictFromFile(file2)
    sim = getSimilarity(dict1,dict2)
    acc = sim*100
    print("Accuracy for ",name1," and ",name2," is : ",acc,"%")


def getTokensDictFromFile(file):
    data = open(file,"rb")
    file_content=' '
    for nffile in data:
        nffile=nffile.decode('utf-8', 'ignore')
        file_content+=nffile
    data=file_content
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
    if(dot_product== 0.0):
        return 0
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



sum_path = summary_path+"/"
grnd_path = ground_path+"/"
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

with open(pickle_path+"/LabelledData.pickle","wb") as pickle_out:
    pickle.dump(usefull_labels,pickle_out)
pickle_out.close()

with open(pickle_path+"/LabelledFreqDict.pickle","wb") as pickle_out:
    pickle.dump(dictFreqWC,pickle_out)
pickle_out.close()
