import os
import math
import pandas

class TfIdf:
    data_set_path = "/home/yogi/Desktop/Webapps Data"                   #Path of data set folder
    data_files = os.listdir(data_set_path)                              #List of all the file names with extension in dataset folder
    file_word_dictionary = {}                                           #Dictionary of files containing various words.
    file_sentence_dictionary ={}                                        #Dictionary of files containing all the sentences
    file_word_frequency = {}                                            #Dictionary of files containing frequency of unique words 
    file_Tf={}                                                          #Dictionary of files containing TF values of the words
    tfIDF = {}                                                          #Dictionary of files containing IDF values of the words
    word_set={}                                                         #It is the union of all the different words in corpus
    
    def dataCleaning(self,word):                                        #Used to clean the words of the file
         word=word.strip('\n,:()"!@#$%^&*=+-.')                         #from various punctuation marks.
         return word
    
    def splitFileToLines(self,sentence_set,file_content):               #Splitting the files into lines
        for line in file_content:                                       #Sentence_set is the dictionary which is initially empty
            sentences = line.split("\n.")                               #file_content is the content of the particular file
            for sentence in sentences:
                sentence=sentence.strip('\n ')
                if(sentence!=''):
                    sentence_set.append(sentence)
        return sentence_set
    
    def splitFileToWords(self,word_set,file_content):                   #Splitting the files into words
        for line in file_content:                                       #word_set is the dictionary which is initially empty
            words = line.split(" ")                                     #file_content is the content of the particular file
            for word in words:
                word=self.dataCleaning(word)
                if(word!=''):
                    word_set.append(word)
        return word_set
    
    def fileRead(self):                                                 #Method to read all the dataset files 
        for file in self.data_files:
            actual_path = self.data_set_path +"/"+ file
            file_content =  open(actual_path,"r")
            file__content =  open(actual_path,"r")
            word_set = []
            sentence_set =[]
            word_set = self.splitFileToWords(word_set,file__content)
            sentence_set = self.splitFileToLines(sentence_set,file_content)
            self.file_word_dictionary[file[:-4]] = word_set
            self.file_sentence_dictionary[file[:-4]] = sentence_set
        return self.file_word_dictionary, self.file_sentence_dictionary
    
    def getWordSet(self):                          #IT gives the collection of all the different words in corpus
        for file_list in self.file_word_dictionary:
            self.word_set= set(self.word_set).union(set(self.file_word_dictionary[file_list]))    
        return(self.word_set)
    
    def computeWordFrequency(self):                #Counting the frequency of words of word collection in a particular file
        for file in self.file_word_dictionary:
            self.file_word_frequency[file+"frequency"] = dict.fromkeys(self.word_set,0)
        for file in self.file_word_dictionary:
            for word in self.file_word_dictionary[file]:
                self.file_word_frequency[file +"frequency"][word] +=1
        return self.file_word_frequency
    
    def computeTF(self,wordDict, BOW):              #Computing TF values  TF = (# of time word appear in document)/ (Total number of words in document)
        tfDict ={}                                  # wordDict : frequency of words in a partiular file, BOW : Total # of words in a particular file
        bow_count =len(BOW)
        for word, count in wordDict.items():
            tfDict[word]=count/float(bow_count)
        return tfDict
    
    def computeIDF(self,documentList):              #Computing IDF values (# of documents)/(# of documents that contain a particular word)
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
    
    def computeTFIDF(self,tfBow,idfs):              #computing TFIDF values (TF(word))*(IDF(word))
        tfidf = {}
        for word, val in tfBow.items():
            tfidf[word] = val * idfs[word]
        return tfidf

    def initiate(self):                             #Main function of the class
        self.file_word_dictionary, self.file_sentence_dictionary=self.fileRead()
        self.word_set = self.getWordSet()
        self.file_word_frequency = self.computeWordFrequency()
        #print(pandas.DataFrame(self.file_word_frequency[file] for file in self.file_word_frequency))
        
        for file in self.file_word_dictionary:                                  #computing TF values
            self.file_Tf[file+"TF"] = self.computeTF(self.file_word_frequency[file+"frequency"], self.file_word_dictionary[file])
        idf = self.computeIDF([self.file_Tf[file] for file in self.file_Tf])    #compute idf value
        for file in self.file_word_dictionary:                                  #computing TFIDF values
            self.tfIDF[file+"tfIDF"] = self.computeTFIDF(self.file_Tf[file+"TF"],idf)
        print(pandas.DataFrame(self.tfIDF[file] for file in self.tfIDF))
        
def main():
    TfIdfObj = TfIdf()
    TfIdfObj.initiate()

if __name__=="__main__":
    main()