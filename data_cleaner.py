#@title Domyślny tekst tytułu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import keras.preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from pandas import Series
import bs4 as bs  
import urllib.request  
import re  
import string
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten



#Data Gathering and variables declaration
#data import and variables declaration
data = pd.read_csv('PPTD.csv', encoding= 'unicode_escape', sep=';')
df=pd.DataFrame(data)
desc1=Series.tolist(df.iloc[:,0])
loc=Series.tolist(df.iloc[:,1])
desc2=Series.tolist(df.iloc[:,2])
category=Series.tolist(df.iloc[:,3])
df['newX']=df['TYTUL1']+ df['Lokalizacja']+df['TYTUL2']
df['newy']=df['CATEGORIA']
lista=[]
finlab=[]
filtered_vocab=[]

poz=0
dictionary=''
Z=len(df)

##definicje funkcji
#==============================================================================
#==============================================================================
def to_list(series_lokalizacja):
  import numpy as np
  import pandas as pd
  from pandas import Series
  #print(" a wchodzi to :", series_lokalizacja)
  series_lokalizacja=df.iloc[:,2]
  series_lokalizacja = Series.tolist(series_lokalizacja.str[1:])
  #print("to wychodzi z series", series_lokalizacja)


  return series_lokalizacja
   
#Linijka ponizej usuwa spacje ktora wystepuje na poczatku kazdej linijki, zakomentuj jak nie potrzebne
    
#Konwersja columny (Series) do listy
    




#===============================================================================
def tok(lista,poz):
  values = array(lista)
  # integer encode
  label_encoder = LabelEncoder()
  integer_encoded = label_encoder.fit_transform(values)
  
  onehot_encoder = OneHotEncoder(sparse=False)
  integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
  onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
  
  edata=onehot_encoded
  ylen=len(edata[:,1])


  
  ylen=int(ylen)
  m=0
  
  while m < ylen:
    #print(values[m])
    mlabel=str(edata[m])
    mlabel=mlabel.replace(".",",")
    df.iloc[m,poz]=mlabel
    m +=1
    finlab.append(mlabel)
  return finlab

#=============================================================================
# Funkcja budowy Bag of words
#=============================================================================
def bagofwords(lista,poz):
  
  values=lista
  arr = values
  values2=[]
  nlista=[]
   
  for x in arr:
    
    x=str(x)
    x=x.split(" ")
    values2.append(x)
  #print(values2)
  tekst=values2
  count='0'
  
  for c in tekst:
    d=''
    #c=c.pop(1)
    #c=c.split("'")
    #print ("to jest c: ",c)
    li=0
    for b in c:
      if b != "]":        
        if b =="pusty]":
          b="pusty"          
          d=b
        else:          
          d += b
          li +=1
          if li != 1:
            d += ' '
    
        #nlista.append(d)
    #df.iloc[?,poz]=d
    nlista.append(d)
    lista=nlista
    return lista
  ##funkcja odbierająca przekonwertowaną listę w celu budowy Bag of words

#========================================================================
#===================================================================
def finbag(values2,poz):
  CountVec = CountVectorizer(ngram_range=(1,1), # to use bigrams ngram_range=(2,2)
                            stop_words='english')
  Count_data = CountVec.fit_transform(values2)
  cv_dataframe=pd.DataFrame(Count_data.toarray(),columns=CountVec.get_feature_names())
  f=0
  while f < Z:
    #y_str1=df.iloc[f,poz]
    y_string=Series.tolist(cv_dataframe.iloc[f,:])
    #y_string=str(y_string)
    #y_string=y_string.replace(",",".")
    df.iloc[f,poz]=y_string
    #df.iloc[f,poz]=Series.tolist(cv_dataframe.iloc[f,:]) 
    
    """y_str2=str(Series.tolist(cv_dataframe.iloc[f,:]))
    print("przed kodowaniem: ", y_str1)
    print("po kodowaniu: ", y_str2)"""

    f+=1
 
  return (cv_dataframe)
 

#==========================================================================
def datacleansing(lista,poz):
  import pandas as pd
  from pandas import Series
  data2 = lista
  data22=[]
  t = 0
  dt=''
  while t < Z:
      input_str=''
      Y_Flag=''
      input_str=data2[t]
      Y_Flag=pd.isna(input_str)
      if Y_Flag == True or input_str =='nan':
        input_str="pusty"
        result=input_str
        df.iloc[t,poz]=result
      else:
        input_str=input_str.lower()
        
        result = re.sub(r'\d+', '', input_str)
        
        #result = result.split(',')
        df.iloc[t,poz]=result
        
      t +=1
  return
#=========================================================================
def my_func(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return arg
#=========================================================================

def vectorize(tokens):
    ''' This function takes list of words in a sentence as input 
    and returns a vector of size of filtered_vocab.It puts 0 if the 
    word is not present in tokens and count of token if present.'''
    vector=[]
    for w in filtered_vocab:
        vector.append(tokens.count(w))
    return vector
def unique(sequence):
    '''This functions returns a list in which the order remains 
    same and no item repeats.Using the set() function does not 
    preserve the original ordering,so i didnt use that instead'''
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]
#============================================================================

# definicje funkcji END
#==============================================================================
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#===========================================================================
# wywołania funkcji
## tokenizacja CATEGORY
lista=category
poz=3
tok(lista,poz)


## tokenizacja Tytul1
lista=desc1
poz=0
tok(lista,poz) 


#___________________________________________________________________________

## tokenizacja lokalizacji
lista=loc
poz=1
datacleansing(lista, poz)    
lista=Series.tolist(df.iloc[:,poz])
#bagofwords(lista,poz)
values2=lista
finbag(values2,poz)

#+==================
#==================== 
## tokenizacja adresu  
lista=desc2
poz=2
datacleansing(lista, poz)   
lista=df.iloc[:,poz]
#bagofwords(lista,poz)
finbag(lista,poz)
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#Inne podejście na zbudowanie vektora wejściowego poprzez połączenie kolumn i wspólny Bag OF Words
#łączę dane
newX=Series.tolist(df['newX'])
lista=newX
poz=4
datacleansing(lista,poz)
newX=Series.tolist(df['newX'])
newy=Series.tolist(df['newy'])
maxlen=100
num_words=10000
embedding_dim=100
tokenizer=Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(newX)
res=list(tokenizer.index_word.items())[:20]
print(res)
sequences=tokenizer.texts_to_sequences(newX)
print("sekwencje",sequences[:5])
word_index=tokenizer.word_index
print(f'{len(word_index)} unikatowych słów')
tokenizer.fit_on_texts(newy)
sequences1=tokenizer.texts_to_sequences(newy)
print("sekwencje",sequences1[:5])
train_data=pad_sequences(sequences, maxlen)
train_data1=pad_sequences(sequences1,maxlen)
X=train_data
y=train_data1
X_ntrain, X_ntest, y_ntrain, y_ntest = train_test_split(
    X, y, test_size=0.33, random_state=0)
"""y_ntrain = np.asarray(y_ntrain).astype('float32').reshape((-1,1))
y_ntest = np.asarray(y_ntest).astype('float32').reshape((-1,1))
X_ntrain = np.asarray(X_ntrain).astype('float32').reshape((-1,1))
X_ntest = np.asarray(X_ntest).astype('float32').reshape((-1,1))'''"""
X_val1 = np.concatenate([X_ntrain, X_ntest])
y_val1=np.concatenate([y_ntrain, y_ntest])
print("X val shape:", X_val1[:30])
print("y val shape:", y_val1.shape)

#=============================================================================
###Budowa sieci neuronowej
####Podział zbioru na Train data i Test data

#X=df[['TYTUL1','Lokalizacja','TYTUL2']]
#X=tf.convert_to_tensor(X, dtype=tf.float32)
X=df['TYTUL2']
y=df['CATEGORIA']


X_data=np.asarray(X)
y_data=np.asarray(y)
xlen=len(X)
'''X_data1=[]

a=0
while a < xlen:
  tensor1=X_data[a,0]
  tensor2=X_data[a,1]
  tensor3=X_data[a,2]


  #tensor1 = my_func(np.array(tensor1))
  #tensor2 = my_func(np.array(tensor2))
  tensor3 = my_func(np.array(tensor3))


  X_data1.append(tensor3)
  y_data1.append(tensor4)

  #print("tensory po konwersji: ",tensor1,tensor2,tensor3)
  #print("odpowiedź: ",tensor4)

  a+=1'''

 


X_train, X_val, y_train, y_val = train_test_split(X_data,y_data, train_size=0.7, random_state=0)
#BATCH_SIZE=32
#model.evaluate(X_ntrain(BATCH_SIZE), steps=None, verbose=1)
####Budowa modelu
model=Sequential()
model.add(Embedding(num_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(100, activation='sigmoid'))


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics='accuracy')
model.summary()

history=model.fit(X_ntrain, y_ntrain, batch_size=32, epochs=50, validation_data=(X_val1,y_val1))

#df.head(5)

def plot_hist(history):
  import pandas as pd
  import plotly.graph_objects as go
  hist=pd.DataFrame(history.history)
  hist['epoch']=history.epoch
  fig=go.Figure()
  fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['accuracy'], name='accuracy', mode='markers+lines'))
  fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_accuracy'], name='val_accuracy', mode='markers+lines'))
  fig.update_layout(width=1000, height=500, title='loss vs val accuracy', xaxis_title='epoki', yaxis_title='accuracy')
  fig.show()

  fig=go.Figure()
  fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['loss'], name='loss', mode='markers+lines'))
  fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_loss'], name='val_loss', mode='markers+lines'))
  fig.update_layout(width=1000, height=500, title='loss vs val accuracy', xaxis_title='epoki', yaxis_title='loss')
  fig.show()


  return

plot_hist(history)