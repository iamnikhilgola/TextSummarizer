import os 
import pandas as pd
import nltk
import gensim
from gensim import corpora,models,similarities
import pickle


path = os.path.abspath(os.path.dirname(__file__))  
pickle_path = os.path.join(path, "Pickled Data")

with open(pickle_path+"/word_Dictionary.pickle","rb") as pickle_in:
    file_word_dictionary = pickle.load(pickle_in)

with open(pickle_path+"/word_Frequency.pickle","rb") as pickle_in:
    file_word_frequency = pickle.load(pickle_in)

with open(pickle_path+"/unique_Wordset.pickle","rb") as pickle_in:
    word_set =pickle.load(pickle_in)
    
df = pd.read_csv('data.csv');

x=df['Document'].values.tolist()
y=df['Data'].values.tolist()

corpus= y

tok_corp = [nltk.word_tokenize(sent) for sent in corpus]

model1 = gensim.models.Word2Vec(tok_corp,min_count=3,size =32)

WordVector ={}
wordings=[]
for files in file_word_dictionary:
    for words in file_word_dictionary[files]:
        wordings.append(words)
        
for w in wordings:
    if model1.wv.__contains__(w):
        WordVector[w]=model1.wv.__getitem__(w)
 
with open(pickle_path+"/word_vector.pickle","wb") as pickle_out:
    pickle.dump(WordVector,pickle_out)
    pickle_out.close()
