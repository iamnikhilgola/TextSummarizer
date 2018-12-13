import pickle
import os
from nltk.collocations import *


path = os.path.abspath(os.path.dirname(__file__))  
pickle_path = os.path.join(path, "Pickled Data")

with open(pickle_path+"/TFIDF_values.pickle","rb") as pickle_in:
    tfIDF = pickle.load(pickle_in)

with open(pickle_path+"/sentence_Dictionary.pickle","rb") as pickle_in:
    file_sentence_dictionary = pickle.load(pickle_in)

with open(pickle_path+"/fileNames.pickle","rb") as pickle_in:
    file_names= pickle.load(pickle_in)
    
with open(pickle_path+"/bigram.pickle","rb") as pickle_in:
    bigrams= pickle.load(pickle_in)

posWord=dict.fromkeys(file_names,[])
commonWord =dict.fromkeys(file_names,0)

for file in tfIDF:
        Array=[]
        count =0
        value =0
        maxx =0
        for i,j in tfIDF[file].items():
            if(j!=0.0 and i!=file.lower()): 
                count+=1
                value+=j
                if(j>maxx):
                    maxx=j
                Array.append((i,j))
        posWord[file] = Array
        commonWord[file] =(count,maxx)#(value/count)*8)
        newWords=dict.fromkeys(file_names,[])
    
for file in posWord:
    nArray=[]
    for x,y in posWord[file]:
        if(y>=commonWord[file][1]):
            nArray.append(x)
    newWords[file]=nArray
for file in newWords:
    print(file,newWords[file])

for file in file_sentence_dictionary:
    summary=[]
    for x in newWords[file]:
        for y in file_sentence_dictionary[file]:
            #print(bigrams[file][0][1])
            if((x and 'privacy') in y):#and bigrams[file][0][0] and bigrams[file][0][1]) in y):
                if y not in summary:
                    summary.append(y)    
    print(file,summary)
    print("\n\n")
#print(bigrams)   
