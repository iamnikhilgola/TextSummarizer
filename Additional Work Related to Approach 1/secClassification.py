import os
import pickle
from nltk import word_tokenize

FileName ="Medium"
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

def computeLabel(word,score):                   #To compute label for a word
    for value in kmCluster:
        if word in kmCluster[value]:
            return value
 
for cluster in kmCluster:
    if 'privacy' in kmCluster[cluster]:
        priC= cluster
    if 'agreement' in kmCluster[cluster]:
        accC = cluster
        
imp_set=[]
data = open("data1.txt","rb")                   #reading self made vocubalary
for nffile in data:
    nffile=nffile.decode('utf-8', 'ignore')
    imp_set.append(nffile.strip("\n\r"))
test_set=[]
#data = open("Happn.txt","rb")    
for nffile in data:
    nffile=nffile.decode('utf-8', 'ignore')
    if nffile!='':
        test_set.append(nffile.strip("\n\r"))

    
imp_set=list(set(imp_set))                      #making set of self made vocubalory
#print(imp_set)

kmCluster[len(kmCluster)]=imp_set                          #making self made words as 9th cluster
        
testData = file_content['Quora']             #taking testdata of facebook

clusterScore={}
for value in kmCluster:                         #giving each cluster a score
     clusterScore[value]=1/len(kmCluster[value])   
     
#for value in kmCluster:
#    print(value,kmCluster[value])
    
uniqueWordC=[]                                  #taking words of the important clusters i.e 4th and 6th
for lab in kmCluster:
    #if lab==2.0 or lab ==6.0:
    for words in kmCluster[lab]:
        uniqueWordC.append(words)
            
dictFreqWC={}                                   #contains cluster value for each sentence
dictLabels ={}                                  #contains nuetralized freq of imp words for each sentence 
#for files in file_sentence_dictionary:          #Reading each file from the corpus
for Datas in file_sentence_dictionary["Quora"]:   #Reading each sentence from the dictionary
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
            #maxvalue=newDict[key]
            #maxkey=key
            #if(key==11.0):
            #    maxx+=newDict[key]*5*clusterScore[key]
            if(key==priC):
                maxkey+=newDict[key]*40*clusterScore[key]
            elif(key==accC):
                maxkey+=newDict[key]*40*clusterScore[key]
            else:
                maxkey+=newDict[key]*1*clusterScore[key]

    #print(maxkey)
    maxkey=int(maxkey)
    if(maxkey>10):
        maxkey=maxkey/10
    dictLabels[Datas]=int(maxkey)                            #currently storing clustering value for each sentence
    if dictLabels[Datas]==priC or dictLabels[Datas]==accC:
        #if Datas not in dictFreqWC:
        dictFreqWC[Datas]=freqDict                  #currently storing nuetralized frequency of each sentence
    #else:
        #dictFreqWC[Datas]=freqDict
           
usefull_labels={}  #assign new labels which correspond to our important marked sentences
useless_words =['please','following','under these terms','payment terms','applicable laws','these terms','under', 'these','report']
privacy=[]       # would store sentences related to privacy
agreement=[]     # would store sentences related to agreement

#for value in dictLabels:
#    print(value,dictLabels[value])
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
    #else:
        #usefull_labels[nm]=3
        #privacy.append(nm)
    #elif dictLabels[nm]==12.0:
    #    agreement.append(nm)
outputFile = open("FacebookSummary.txt","w")
for values in privacy:
    print(values)
    outputFile.write(values+"\n")
outputFile.close()
#for line in privacy:
#    print(line)

#print(agreement)
#print(dictFreqWC)
with open(pickle_path+"/LabelledData.pickle","wb") as pickle_out:
    pickle.dump(usefull_labels,pickle_out)
pickle_out.close()

with open(pickle_path+"/LabelledFreqDict.pickle","wb") as pickle_out:
    pickle.dump(dictFreqWC,pickle_out)
pickle_out.close()
