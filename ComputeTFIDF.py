import math
import pickle
import os

path = os.path.abspath(os.path.dirname(__file__))  
pickle_path = os.path.join(path, "Pickled Data")

with open(pickle_path+"/word_Dictionary.pickle","rb") as pickle_in:
    file_word_dictionary = pickle.load(pickle_in)

with open(pickle_path+"/word_Frequency.pickle","rb") as pickle_in:
    file_word_frequency = pickle.load(pickle_in)



def computeTF(wordDict, BOW):              #Computing TF values  TF = (# of time word appear in document)/ (Total number of words in document)
    tfDict ={}                                  # wordDict : frequency of words in a partiular file, BOW : Total # of words in a particular file
    bow_count =len(BOW)
    for word, count in wordDict.items():
        tfDict[word]=count/float(bow_count)
    return tfDict

def computeIDF(documentList):              #Computing IDF values (# of documents)/(# of documents that contain a particular word)
    idfDict = {}                                #documentList : list of idf values of all documents
    length = len(documentList)
    idfDict = dict.fromkeys(documentList[1].keys(),0)   #documentList[1] becaue we need just keys n all keys are same
    for doc in documentList:
        for word,value in doc.items():
            if(value>0):
                idfDict[word] +=1
    for word, val in idfDict.items():
        idfDict[word] = math.log(length/float(val))
    return idfDict

def computeTFIDF(tfBow,idfs):              #computing TFIDF values (TF(word))*(IDF(word))
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val * idfs[word]
    return tfidf

def main():
    file_Tf={}                                                          #Dictionary of files containing TF values of the words
    tfIDF = {}                                                          #Dictionary of files containing IDF values of the words
    for file in file_word_dictionary:                                  #computing TF values
        file_Tf[file] = computeTF(file_word_frequency[file], file_word_dictionary[file])
    idf = computeIDF([file_Tf[file] for file in file_Tf])               #compute idf value
    
    for file in file_word_dictionary:                                  #computing TFIDF values
        tfIDF[file] = computeTFIDF(file_Tf[file],idf)
        #print(tfIDF)
    
    with open(pickle_path+"/TFIDF_values.pickle","wb") as pickle_out:
        pickle.dump(tfIDF,pickle_out)
    pickle_out.close()    
    
    with open(pickle_path+"/TF_values.pickle","wb") as pickle_out:
        pickle.dump(file_Tf,pickle_out)
    pickle_out.close()
    
if __name__=="__main__":
    main()