# =============================================================================
# This File is responsible for Doing Preprocessing
# 
# =============================================================================

import os
import re
import copy
import time
import pickle
import nltk
import csv
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.tokenize import sent_tokenize
from nltk import ngrams
from nltk import pos_tag


class PreProcessing:
    def __init__(self):
        self.path = os.path.abspath(os.path.dirname(__file__))                   #Path of data set folder
        self.data_set_path = os.path.join(self.path, "Datasets/Webapps Data")
        if not (os.path.isdir("Pickled Data")):
            os.mkdir('Pickled Data')
        self.pickle_path = os.path.join(self.path, "Pickled Data")
        self.test_file_path = os.path.join(self.path, "TestFiles")
        self.data_test_file =os.listdir(self.test_file_path)
        self.data_files = os.listdir(self.data_set_path)                         #List of all the file names with extension in dataset folder
        self.file_word_dictionary = {}                                           #Dictionary of files containing various words.
        self.file_sentence_dictionary ={}                                        #Dictionary of files containing all the sentences
        self.file_word_frequency = {}                                            #Dictionary of files containing frequency of unique words 
        self.word_set={}                                                         #It is the union of all the different words in corpus
        self.stop_words =set(stopwords.words("english"))                         #It contains all the stop words in english language
        self.file_names =[]                                                      #Contain File names without extension
        self.file_bigram_dictionary ={}                                          #Contain all bigrams (word appear 2 times)
        self.file_pos_tag ={}                                                    #Contains tags for each file   
        self.file_content = {}
        self.file_ngram_dictionary ={} 
        
    def setFileName(self):
        for name in self.data_files:
            self.file_names.append(name[:-4])
            
    def dataCleaning(self,word):                                        #Used to clean the words of the file
         word=word.strip('\n,:()"!@#$%^&*=+-.\'1234567890;:<>/')                         #from various punctuation marks.
         return word
    
    def hasNumbers(self,inputString):
        return bool(re.search(r'\d', inputString))
    
    def getPosTags(self,tag):
        if tag.startswith('J'):		# adjective
            return 'a'
        elif tag.startswith('V'):	# verb
            return 'v'
        elif tag.startswith('N'):	# noun
            return 'n'
        elif tag.startswith('R'):	# adverb
            return 'r'
        else:
            return ''

    def getBigrams(self,word_set):
        bigrams=list(ngrams(word_set,2))
        return bigrams
    
    def nGrams(self,word_set,len):
        nngrams = ngrams(word_set,len)
        return nngrams
    
    def splitFileToLines(self,sentence_set,file_content):               #Splitting the files into lines
        for line in file_content.splitlines():                          #Sentence_set is the dictionary which is initially empty
            sentences = line.split("\n")                               #file_content is the content of the particular file
            for sentence in sentences:
                sentence=sentence.strip('\n ')
                sentence= re.sub(r"http\S+", '', sentence, flags=re.MULTILINE)
                if not self.hasNumbers(sentence):
                    if(sentence!=''):
                        sentence_set.append(sentence.lower()) 
        return sentence_set
    
    def splitFileToWords(self,word_set,file_content):                   #Splitting the files into words
        newTag=[]
        WN=WordNetLemmatizer()
        for line in file_content.splitlines():                          #word_set is the dictionary which is initially empty
            words = line.split(" ")                                     #file_content is the content of the particular file
            for word in words:
                word=word.lower()
                #word=self.dataCleaning(word)                           #function not required as isalpha used below
                if word not in self.stop_words:                         #checking if word not in stop word
                    word=WN.lemmatize(word)
                    if(word!=''and len(word)>4 and word.isalpha()):     #isalpha to filter punctuation
                        word_set.append(word)
        
        newTag=pos_tag(word_set)
        lemmatizedWords=[]
        for i in range(len(word_set)):                      #Lemmetizing the word file
            if(self.getPosTags(newTag[i][1]) == ''):
                lemma= WN.lemmatize(word_set[i])
            elif self.getPosTags(newTag[i][1]) == 'r' and word_set[i].endswith('ly'):
                lemma = word_set[i].replace('ly','')
            else:
                lemma = WN.lemmatize(word_set[i], pos=self.getPosTags(newTag[i][1]))
            lemmatizedWords.append(lemma)
        return lemmatizedWords,newTag
    
    def fileRead(self):                                                 #Method to read all the dataset files 
        for file in self.data_files:
            actual_path = self.data_set_path +"/"+ file
            new_file__content =  open(actual_path,"rb")
            file_content=' '
            for nffile in new_file__content:
                nffile=nffile.decode('utf-8', 'ignore')
                file_content+=nffile
            self.file_content[file[:-4]] =file_content
            file__content=copy.deepcopy(file_content)
            word_set = []
            sentence_set =[]
            bigram_set =[]
            n_gram_set =[]
            tags=[]
            word_set,newTag= self.splitFileToWords(word_set,file__content)
            tags+=newTag
            sentence_set = self.splitFileToLines(sentence_set,file_content)
            finder = BigramCollocationFinder.from_words(word_set, window_size = 3)
            bigram_set=finder.nbest(BigramAssocMeasures.likelihood_ratio,1)
            n_gram_set=self.nGrams(word_set,8)
            new_set=[list(value) for value in n_gram_set]
            tags=pos_tag(word_set)
            self.file_sentence_dictionary[file[:-4]] = sentence_set
            self.file_word_dictionary[file[:-4]] = word_set
            self.file_bigram_dictionary[file[:-4]] = bigram_set
            self.file_ngram_dictionary[file[:-4]]=new_set
            self.file_pos_tag[file[:-4]]=tags
        return self.file_word_dictionary, self.file_sentence_dictionary,self.file_bigram_dictionary,self.file_pos_tag
    
    def getWordSet(self):                          #IT gives the collection of all the different words in corpus
        for file_list in self.file_word_dictionary:
            self.word_set= set(self.word_set).union(set(self.file_word_dictionary[file_list]))    
        return(self.word_set)
    
    def computeWordFrequency(self):                #Counting the frequency of words of word collection in a particular file
        for file in self.file_word_dictionary:
            self.file_word_frequency[file] = dict.fromkeys(self.word_set,0)
        for file in self.file_word_dictionary:
            for word in self.file_word_dictionary[file]:
                self.file_word_frequency[file][word] +=1
        return self.file_word_frequency
    
    def fileTestRead(self):                                                 #Method to read all the dataset files 
        file_sentence_dictionary={}
        for file in self.data_test_file:
                actual_path = self.test_file_path +"/"+ file
                new_file__content =  open(actual_path,"rb")
                file_content=' '
                for nffile in new_file__content:
                    nffile=nffile.decode('utf-8', 'ignore')
                    file_content+=nffile
                    #file_content[file[:-4]] =file_content
                sentence_set =[]
                sentence_set = self.splitFileToLines(sentence_set,file_content)
                file_sentence_dictionary[file[:-4]] = sentence_set
        return file_sentence_dictionary
    
    def initiate(self):                             #Main function of the class
        self.file_word_dictionary, self.file_sentence_dictionary,self.file_bigram_dictionary,self.file_pos_tag=self.fileRead()
        self.word_set = self.getWordSet()
        self.file_word_frequency = self.computeWordFrequency()
        self.setFileName()
        TestfileData={}
        TestfileData =self.fileTestRead()
        
        with open('data.csv', 'w') as csvfile:
            fieldnames = ['Document', 'Data']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for file in self.file_content:
                writer.writerow({'Document':file,'Data':self.file_content[file]})

        with open(self.pickle_path+"/test_dictionary.pickle","wb") as pickle_out:
            pickle.dump(TestfileData,pickle_out)
        pickle_out.close()
        
        with open(self.pickle_path+"/word_Dictionary.pickle","wb") as pickle_out:
            pickle.dump(self.file_word_dictionary,pickle_out)
        pickle_out.close()
        
        with open(self.pickle_path+"/word_Frequency.pickle","wb") as pickle_out:
            pickle.dump(self.file_word_frequency,pickle_out)
        pickle_out.close()
        
        with open(self.pickle_path+"/sentence_Dictionary.pickle","wb") as pickle_out:
            pickle.dump(self.file_sentence_dictionary,pickle_out)
        pickle_out.close()
        
        with open(self.pickle_path+"/fileNames.pickle","wb") as pickle_out:
            pickle.dump(self.file_names,pickle_out)
        pickle_out.close()
        
        with open(self.pickle_path+"/unique_Wordset.pickle","wb") as pickle_out:
            pickle.dump(self.word_set,pickle_out)
        pickle_out.close()
        
        with open(self.pickle_path+"/bigram.pickle","wb") as pickle_out:
            pickle.dump(self.file_bigram_dictionary,pickle_out)
        pickle_out.close()
        
        with open(self.pickle_path+"/file_content.pickle","wb") as pickle_out:
            pickle.dump(self.file_content,pickle_out)
        pickle_out.close()
        
        with open(self.pickle_path+"/ngram.pickle","wb") as pickle_out:
            pickle.dump(self.file_ngram_dictionary,pickle_out)
        pickle_out.close()
                
                        
def main():
    PreprocessObj = PreProcessing()
    PreprocessObj.initiate()

if __name__=="__main__":
    start_time = time.time()
    main()
    end_time = time.time()