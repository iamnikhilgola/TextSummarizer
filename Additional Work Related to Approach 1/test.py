import os
import re
import pickle
import fileinput
from nltk import word_tokenize

path = os.path.abspath(os.path.dirname(__file__))  
pickle_path = os.path.join(path, "Pickled Data")

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


def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

def splitFileToLines(sentence_set,file_content):               #Splitting the files into lines
    for line in file_content:                                      #Sentence_set is the dictionary which is initially empty
        sentences = line.split("\n.")                               #file_content is the content of the particular file
        for sentence in sentences:
            sentence=sentence.strip('\n ')
            sentence= re.sub(r"http\S+", '', sentence, flags=re.MULTILINE)
            if not hasNumbers(sentence):
                if(sentence!=''):
                    sentence_set.append(sentence.lower()) 
    return sentence_set

def computeLabel(word,score):                   #To compute label for a word
    for value in kmCluster:
        if word in kmCluster[value]:
            return value
        
imp_set=[]
data = open("data1.txt","rb")                   #reading self made vocubalary
for nffile in data:
    nffile=nffile.decode('utf-8', 'ignore')
    imp_set.append(nffile.strip("\n\r"))

        
test_set=[]
data = open("Happn.txt","rb")                   #reading self made vocubalary
for nffile in data:
    nffile=nffile.decode('utf-8', 'ignore')
    test_set.append(nffile.strip("\n\r"))
sentence_set=[]
sentence_set=splitFileToLines(sentence_set,test_set)

imp_set=list(set(imp_set))                      #making set of self made vocubalory
#print(imp_set)

kmCluster[8.0]=imp_set                          #making self made words as 9th cluster
        
testData = file_content['Quora']                #taking testdata of facebook

clusterScore={}
for value in kmCluster:                         #giving each cluster a score
     clusterScore[value]=1/len(kmCluster[value])   
     
uniqueWordC=[]                                  #taking words of the important clusters i.e 4th and 6th
for lab in kmCluster:
    #if lab==2.0 or lab ==6.0:
    for words in kmCluster[lab]:
        uniqueWordC.append(words)

for lab in kmCluster:
    print(lab,kmCluster[lab])
            
dictFreqWC={}                                   #contains cluster value for each sentence
dictLabels ={}                                  #contains nuetralized freq of imp words for each sentence 


count =0
for Datas in file_sentence_dictionary['DuckDuckGo']:   #Reading each sentence from the dictionary
    #print(count,Datas)
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
            if(key==2.0):
                maxkey+=newDict[key]*40*clusterScore[key]
            elif(key==6.0):
                maxkey+=newDict[key]*40*clusterScore[key]
            else:
                maxkey+=newDict[key]*1*clusterScore[key]
    count+=1
    maxkey=int(maxkey)
    if(maxkey>10):
        maxkey=maxkey/10
    dictLabels[Datas]=int(maxkey)                            #currently storing clustering value for each sentence
    if dictLabels[Datas]==2.0 or dictLabels[Datas]==6.0:
        #if Datas not in dictFreqWC:
        dictFreqWC[Datas]=freqDict                  #currently storing nuetralized frequency of each sentence
    #else:
        #dictFreqWC[Datas]=freqDict
           
usefull_labels={}  #assign new labels which correspond to our important marked sentences
useless_words =['please','following','under these terms','payment terms','applicable laws','these terms','under', 'these','report']
privacy=[]       # would store sentences related to privacy
agreement=[]     # would store sentences related to agreement

for nm in dictLabels:    #Taking each sentence from our labelled sentence
    flag=0
    if dictLabels[nm]==2.0 :
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
    if dictLabels[nm]==6.0 :       
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
            #privacy.append(nm)           
    else:
        usefull_labels[nm]=3
        privacy.append(nm)
        
print(privacy)
outputFile = open("DuckDuckGosummary.txt","w")
for values in privacy:
    outputFile.write(values+"\n")
outputFile.close()

with open(pickle_path+"/LabelledData.pickle","wb") as pickle_out:
    pickle.dump(usefull_labels,pickle_out)
pickle_out.close()

with open(pickle_path+"/LabelledFreqDict.pickle","wb") as pickle_out:
    pickle.dump(dictFreqWC,pickle_out)
pickle_out.close()