#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
import string
import re
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from keras.layers import Dropout, Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer

import pickle
import warnings
import logging
import tensorflow as tf
logging.basicConfig(level=logging.INFO)

import transformers
from transformers import   RobertaTokenizer, TFRobertaModel


#Flask API

from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
import json
import ast


# In[2]:


english_punctuations = string.punctuation
punctuations_list = english_punctuations
additional_stopwords = ['rt','vs','covid19nnchernobyl', 'httpstcob7qi0shk7g','httpstcoirnjm1xvmy', 'covid19','abscbnnews','httpstcojawhu','outbluerst','covidph', 'httpstcozxjgaxblya', '2248071','20745','69','#covid19ph','rohtak','jind','sirsa','sonipat','faridabad','indiafightscorona','kurukshetra','gurgaon','covid19india','haryana', 'httpstcotpuiehnaga', 'httpstcocxcybcjjzb', 'duquen', 'deybest','n','tabogo', 'deybesta','tutu', 'estafa','httpstcfzmqvjhd' ]
collection_words = ['#covidph','#covid19ph','#coronavirus', '#covidph','#DOH', '#coronaph','#qcprotektodo','#gcq','resbakuna', 'social distancing','delta variant','lockdown','bakuna', '#Halalan2022', '#LeniRobredoForPresident']
tl_stopwords = ["akin","aking","ako","alin","am","amin","aming","ang","ano","anumang","apat","at","atin","ating","ay","bababa","bago","bakit","bawat","bilang","dahil","dalawa","dapat","din","dito","doon","gagawin","gayunman","ginagawa","ginawa","ginawang","gumawa","gusto","habang","hanggang","hindi","huwag","iba","ibaba","ibabaw","ibig","ikaw","ilagay","ilalim","ilan","inyong","isa","isang","itaas","ito","iyo","iyon","iyong","ka","kahit","kailangan","kailanman","kami","kanila","kanilang","kanino","kanya","kanyang","kapag","kapwa","karamihan","katiyakan","katulad","kaya","kaysa","ko","kong","kulang","kumuha","kung","laban","lahat","lamang","likod","lima","maaari","maaaring","maging","mahusay","makita","marami","marapat","masyado","may","mayroon","mga","minsan","mismo","mula","muli","na","nabanggit","naging","nagkaroon","nais","nakita","namin","napaka","narito","nasaan","ng","ngayon","ni","nila","nilang","nito","niya","niyang","noon","o","pa","paano","pababa","paggawa","pagitan","pagkakaroon","pagkatapos","palabas","pamamagitan","panahon","pangalawa","para","paraan","pareho","pataas","pero","pumunta","pumupunta","sa","saan","sabi","sabihin","sarili","sila","sino","siya","tatlo","tayo","tulad","tungkol","una","walang", "po", "mas","pang","lang","si","kay","ba","mo","naman","di","ba","yung","nga","kayo","yan", "anona", "anuna", "tang", "buking", "mag"]

stopwords = nltk.corpus.stopwords.words('english') + additional_stopwords +tl_stopwords + collection_words


# In[3]:



def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stopwords])

def remove_hashmentions(text):
    clean_tweet = re.sub('(@[a-z0-9]+)\w+',' ', text)
    clean_tweet = re.sub("#[A-Za-z0-9_]+"," ", clean_tweet)
    return clean_tweet

def cleaning_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)

def remove_repeating_chars(text):
    return re.sub(r'(.)1+', r'1', text)

def remove_URLs(text):
    return re.sub('((www.[^s]+)|(https?://[^s]+))','',text)

def remove_numbers(text):
    return re.sub('[0-9]+', '', text)

def preprocess(data):
    if input is None:
        raise Exception("No text provided")
    df = pd.DataFrame(columns =['Tweet'])
    df = df.append({'Tweet' : data}, ignore_index = True)
    
    
    #preprocess steps
    #change to lowercase
    df['Tweet'] = df['Tweet'].apply(lambda x: x.lower())
    #remove stopwords
    df["Tweet"] = df["Tweet"].apply(lambda text: remove_stopwords(text))
    #remove hashtags and mentions
    df['Tweet'] = df['Tweet'].apply(lambda x: remove_hashmentions(x))
    #remove punctuations
    df['Tweet']= df['Tweet'].apply(lambda x: cleaning_punctuations(x))
    #remove repeating characters
    df['Tweet'] = df['Tweet'].apply(lambda x: remove_repeating_chars(x))
    #remove urls
    df['Tweet'] = df['Tweet'].apply(lambda x: remove_URLs(x))    
    #remove numbers
    df['Tweet'] = df['Tweet'].apply(lambda x: remove_numbers(x))
    
    return df.Tweet[0]


# In[4]:


MAX_LEN = 280
MODEL_NAME = "roberta-base"

def roberta_encode(texts, tokenizer):
    ct = len(texts)
    input_ids = np.ones((ct, MAX_LEN), dtype='int32')
    attention_mask = np.zeros((ct, MAX_LEN), dtype='int32')
    token_type_ids = np.zeros((ct, MAX_LEN), dtype='int32') # Not used in text classification

    for k, text in enumerate(texts):
        # Tokenize
        tok_text = tokenizer.tokenize(text)
        
        # Truncate and convert tokens to numerical IDs
        enc_text = tokenizer.convert_tokens_to_ids(tok_text[:(MAX_LEN-2)])
        
        input_length = len(enc_text) + 2
        input_length = input_length if input_length < MAX_LEN else MAX_LEN
        
        # Add tokens [CLS] and [SEP] at the beginning and the end
        input_ids[k,:input_length] = np.asarray([0] + enc_text + [2], dtype='int32')
        
        # Set to 1s in the attention input
        attention_mask[k,:input_length] = 1

    return {
        'input_word_ids': input_ids,
        'input_mask': attention_mask,
        'input_type_ids': token_type_ids
    }

def build_model(n_categories):
        input_word_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_word_ids')
        input_mask = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_mask')
        input_type_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_type_ids')

        # Import RoBERTa model from HuggingFace
        roberta_model = TFRobertaModel.from_pretrained(MODEL_NAME)
        x = roberta_model(input_word_ids, attention_mask=input_mask, token_type_ids=input_type_ids)

        # Huggingface transformers have multiple outputs, embeddings are the first one,
        # so let's slice out the first position
        x = x[0]

        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(n_categories, activation='softmax')(x)

        model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=x)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

        return model



#Trust Model
Roberta_Trust = './roberta-base.h5'
#Tokenizer

#Polarity Model
tfidf_model = 'tf_idf.pk'
polarity_model = 'polarity_model.sav'



app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('data')


# In[7]:


class TrustClassifier(Resource):
    def post(self):
        args = parser.parse_args()  # parse arguments to dictionary
        input = args['data']
        preprocessed = preprocess(input)
        result = []
        
        #Trust Category
        trust_features = roberta_encode([input], tokenizer)
        trust_predictions = model.predict(trust_features)
        trust_pred = np.argmax(trust_predictions, axis = 1)
        output_trust = {0: 'Institutional', 1: 'Behavioral', 2: 'Operational'}
        
        
        #Polarity 

        pol_features = loaded_tfidf.transform([preprocessed])
        pol_pred = loaded_model.predict(pol_features)
        output_polarity ={0: 'Negative', 1: 'Positive'}
        
        
        result.append(output_trust[trust_pred[0]])
        result.append(output_polarity[pol_pred[0]])
        return result




api.add_resource(TrustClassifier, '/trust')




if __name__ == '__main__':
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base",) #Tokenizer
    model = build_model(3)
    model.load_weights(Roberta_Trust)
    loaded_model = pickle.load(open(polarity_model, 'rb'))
    loaded_tfidf = pickle.load(open(tfidf_model, 'rb'))
    app.run(debug=False, port=8080)

