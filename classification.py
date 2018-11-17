# =============================================================================
# This code is reponsible for Doing Classification 
# =============================================================================

import os
import pickle
import operator
import sklearn
import pandas as pd
from sklearn.naive_bayes import GaussianNB
#from sklearn.cross_validation import train_test_split
#from sklearn import matrics
#from sklearn.metrics import accuracy_score
from nltk import word_tokenize

path = os.path.abspath(os.path.dirname(__file__))  
pickle_path = os.path.join(path, "Pickled Data")

with open(pickle_path+"/LabelledData.pickle","rb") as pickle_in:
    labelData = pickle.load(pickle_in)
   
with open(pickle_path+"/LabelledFreqDict.pickle","rb") as pickle_in:
    labelFreqDict = pickle.load(pickle_in)
 
with open(pickle_path+"/K_MeansCluster.pickle","rb") as pickle_in:
    kmCluster = pickle.load(pickle_in)

testData=[]
data = open("Instagram.txt","rb")
for nffile in data:
    nffile=nffile.decode('utf-8', 'ignore')
    if(nffile.strip("\n")!=''):
        nffile=nffile.strip("\n")
        testData.append(nffile)

uniqueWordC=[]
for lab in kmCluster:
    if lab==6.0 or lab ==4.0:
        for words in kmCluster[lab]:
            uniqueWordC.append(words)

WC={}
for Datas in testData:
    Data=word_tokenize(Datas)
    freqDict={}
    for uwc in uniqueWordC:
        freqDict[uwc]=1
    for word in Data:
        if word in uniqueWordC:                 #Assigning count of particular word
            freqDict[word]+=1
    leng=len(Datas)                           #lenght of the Sentence
    for uwc in uniqueWordC:
        freqDict[uwc]/=leng
    WC[Datas]=freqDict

testDF = pd.DataFrame.from_dict(WC,orient='index')
#print(testDF)   


labelDataList = []
for value in labelData:
    labelDataList.append(value)

print(labelData)
labelDF = pd.DataFrame.from_dict(labelData,orient='index')
print(labelDF)
newDF=pd.DataFrame.from_dict(labelFreqDict,orient='index')
#print(newDF)
#mergedDF=labelDF.join(newDF)
#mergedDF=pd.merge(labelDF,newDF,how="inner",on="index")

mergedDF=pd.concat([labelDF, newDF], axis=1,sort='False')

print(mergedDF)
GaussNB = GaussianNB()
GaussNB.fit(mergedDF.values[:,1:],mergedDF.values[:,0])

predicted = GaussNB.predict(testDF.values[:,0:])
print(predicted)

summary=[]
count=0
for value in predicted:
    if value ==1 or value ==2:
        summary.append(testData[count])
print(summary)
#labelFreqDF = pd.DataFrame(labelFreqDict)
#mergedDF=labelDF.merge(labelFreqDF ,left_index=True, right_index=True, how='inner')
#for sentence in labelFreqDict:
#    print(sentence,labelFreqDict[sentence])
#for sentences in labelData:
#    if(labelData[sentences]==1 or labelData[sentences]==2):
#        print(sentences,labelData[sentences])