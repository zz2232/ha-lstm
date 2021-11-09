import pandas as pd
import numpy as np
np.random.seed(1337)

import tensorflow as tf
from tensorflow.keras import layers, optimizers, activations, Model, initializers, utils, Input
from tensorflow.keras.layers import Activation, Dense, Flatten, Lambda, Layer, InputSpec, Dropout
from tensorflow.keras.optimizers import Adam

import nltk
import re
from nltk.corpus import stopwords
from pandas.core.frame import DataFrame
from tensorflow.keras.utils import to_categorical
import gensim
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_sequence_length = 19
class_num = 6

def clean_str(string):
    string = re.sub(r"[`'\",.!?()]", " ", string)
    string = re.sub(r"[^A-Za-z0-9:(),!?\'\`]", " ", string)
    string = re.sub(r":", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"[0-9]", "", string) 
    string = re.sub(r'\b\w\b', '', string) 
    string = re.sub(r' +', ' ', string) 
    return string.strip().lower()

data_set = list(open('/train_5500.label', encoding='utf-8', errors='replace').readlines())
data_set_cleaned = [clean_str(sent) for sent in data_set]
Y_Train = [s.split(' ')[0].split(':')[0] for s in data_set_cleaned]

df_label = DataFrame(Y_Train)
df_label.columns = ['label']
df_label.loc[(df_label.label == 'enty'),'label'] = 0
df_label.loc[(df_label.label == 'hum'),'label'] = 1
df_label.loc[(df_label.label == 'desc'),'label'] = 2
df_label.loc[(df_label.label == 'num'),'label'] = 3
df_label.loc[(df_label.label == 'loc'),'label'] = 4
df_label.loc[(df_label.label == 'abbr'),'label'] = 5

df_text = DataFrame(data_set_cleaned)
df_text.columns = ['text']
df = pd.concat([df_label, df_text], axis = 1) 
total_labels = df["label"].value_counts() 

stop_words = stopwords.words("english")
new_stopwords = ['enty','hum','desc','num','loc','abbr']
stop_words.extend(new_stopwords)
stop_words = [word.replace("\'", "") for word in stop_words]
remove_stop_words = lambda row: " ".join([token for token in row.split(" ") \
                                          if token not in stop_words])
df["preprocessed"] = df["text"].apply(remove_stop_words)
df['totalwords'] = df['preprocessed'].str.split().str.len()

texts = []
labels = []
for i in range(len(df)):
    texts.append(df.iloc[i].preprocessed) 
    labels.append(df.iloc[i].label) 
trick = np.asarray(labels)
labels = to_categorical(trick, num_classes=class_num)
f = list(open('/glove.6B.100d.txt', encoding='utf-8').readlines())

vocab_list = [s.split(' ')[0] for s in f]
vocab_embed_list = [s.split(' ')[1:] for s in f]
embedding_dim = len(f[0].split(' ')[1:]) 
word_index = {" ": 0}   
word_vector = {}   
embeddings_matrix = np.zeros((len(vocab_list) + 1, embedding_dim))
for i in range(len(vocab_list)):
    word = vocab_list[i]  
    word_index[word] = i + 1 
    word_vector[word] = vocab_embed_list[i] 
    embeddings_matrix[i + 1] = vocab_embed_list[i]  

data = []
for sentence in texts:
    new_txt = []
    sentence_splited = sentence.split()
    for word in sentence_splited:
        try:
            new_txt.append(word_index[word])  
        except:
            new_txt.append(0)  
            
    data.append(new_txt)

texts = pad_sequences(data, maxlen = max_sequence_length)
embedding_len = embedding_dim 
embedding_layer = Embedding(input_dim = len(embeddings_matrix), 
                            output_dim = embedding_len, 
                            weights=[embeddings_matrix], 
                            input_length=max_sequence_length, 
                            trainable=False, 
                            name= 'embedding_layer'
                            )

from ha_lstm import MYLSTM_SF
# =============================================================================
# The MYLSTM_SF denotes the HA-LSTM structure in the code and N=4 in this case.
# =============================================================================
sequence_input = Input(shape=(max_sequence_length,))
embedded_sequences = embedding_layer(sequence_input)

x = MYLSTM_SF(units=128)(embedded_sequences)
x = Flatten()(x)
x = Dense(units=32)(x)
x = Dropout(rate=0.1)(x)
x = Dense(units=class_num)(x)
x = Activation('softmax')(x)

model = Model(sequence_input, x)
model.compile(optimizer=Adam(learning_rate=0.0006), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x=texts, y=labels, epochs=50, batch_size=120, verbose=1)