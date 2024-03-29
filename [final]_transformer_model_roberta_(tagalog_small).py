# -*- coding: utf-8 -*-
"""[FINAL] Transformer Model- Roberta (tagalog-small)

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1X3d0igt6JjoqP5LBceLPtd3wPMYAQYFW
"""

!pip install pycaret

pip install transformers

pip install tensorflow

pip install keras

from google.colab import files
uploaded = files.upload()

from google.colab import drive
drive.mount('/content/drive')

"""**Imports**"""

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

from keras.layers import Dropout, Dense, Embedding, LSTM, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils import shuffle
import pickle
import warnings
import logging
import tensorflow as tf
logging.basicConfig(level=logging.INFO)

import transformers
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification,TFRobertaForSequenceClassification, TFRobertaModel

from pycaret.utils import enable_colab
enable_colab()

"""**Load Data**"""

import io
df = pd.read_csv(io.StringIO(uploaded['translated_tagged_tweets.csv'].decode('latin-1')))
print(df)

df.dropna(subset = ["Combined"], inplace=True)

df.value_counts(df["Combined"])

## remove filipino dialect tweets
df = df[df.Combined != "Filipino Dialects"]
df = df[df.Combined != "Irrelevant"]
df = df.reset_index()

df

df.value_counts(df["Combined"])

col = ["tweet_trans","Combined"]
df = df[col]

df.columns

df.columns = ['Tweet', 'Label']

df

df.drop_duplicates(subset= 'Tweet', inplace = True)
# df = df.drop_duplicates(keep='first', subset=['Tweet'])
# df = df.reset_index()
# df = df.rename(columns={'index':'orig_index'})
df

df.loc[df["Label"] == "Institutional-Negative", "Label"] = "Institutional"
df.loc[df["Label"] == "Institutional-Positive", "Label"] = "Institutional"
df.loc[df["Label"] == "Behavioral-Negative", "Label"] = "Behavioral"
df.loc[df["Label"] == "Behavioral-Positive", "Label"] = "Behavioral"
df.loc[df["Label"] == "Operational-Negative", "Label"] = "Operational"
df.loc[df["Label"] == "Operational-Positive", "Label"] = "Operational"

df['category_id'] = df['Label'].factorize()[0]
from io import StringIO
category_id_df = df[['Label', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Label']].values)

df

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
df.groupby('Label').Tweet.count().plot.bar(ylim=0)
plt.show()

df.value_counts(df["Label"])

"""## **Data Preprocessing**

**Making Tweets in lower case**
"""

df["Tweet"] = df["Tweet"].str.lower()
df

"""**Cleaning and stopword removal**"""

additional_stopwords = ['rt','vs','covid19nnchernobyl', 'httpstcob7qi0shk7g','httpstcoirnjm1xvmy', 'covid19','abscbnnews','httpstcojawhu','outbluerst','covidph', 'httpstcozxjgaxblya', '2248071','20745','69','#covid19ph','rohtak','jind','sirsa','sonipat','faridabad','indiafightscorona','kurukshetra','gurgaon','covid19india','haryana', 'httpstcotpuiehnaga', 'httpstcocxcybcjjzb', 'duquen', 'deybest','n','tabogo', 'deybesta','tutu', 'estafa','httpstcfzmqvjhd' ]
collection_words = ['#covidph','#covid19ph','#coronavirus', '#covidph','#DOH', '#coronaph','#qcprotektodo','#gcq','resbakuna', 'social distancing','delta variant','lockdown','bakuna', '#Halalan2022', '#LeniRobredoForPresident']
tl_stopwords = ["akin","aking","ako","alin","am","amin","aming","ang","ano","anumang","apat","at","atin","ating","ay","bababa","bago","bakit","bawat","bilang","dahil","dalawa","dapat","din","dito","doon","gagawin","gayunman","ginagawa","ginawa","ginawang","gumawa","gusto","habang","hanggang","hindi","huwag","iba","ibaba","ibabaw","ibig","ikaw","ilagay","ilalim","ilan","inyong","isa","isang","itaas","ito","iyo","iyon","iyong","ka","kahit","kailangan","kailanman","kami","kanila","kanilang","kanino","kanya","kanyang","kapag","kapwa","karamihan","katiyakan","katulad","kaya","kaysa","ko","kong","kulang","kumuha","kung","laban","lahat","lamang","likod","lima","maaari","maaaring","maging","mahusay","makita","marami","marapat","masyado","may","mayroon","mga","minsan","mismo","mula","muli","na","nabanggit","naging","nagkaroon","nais","nakita","namin","napaka","narito","nasaan","ng","ngayon","ni","nila","nilang","nito","niya","niyang","noon","o","pa","paano","pababa","paggawa","pagitan","pagkakaroon","pagkatapos","palabas","pamamagitan","panahon","pangalawa","para","paraan","pareho","pataas","pero","pumunta","pumupunta","sa","saan","sabi","sabihin","sarili","sila","sino","siya","tatlo","tayo","tulad","tungkol","una","walang", "po", "mas","pang","lang","si","kay","ba","mo","naman","di","ba","yung","nga","kayo","yan", "anona", "anuna", "tang", "buking", "mag"]

stopwords = nltk.corpus.stopwords.words('english') + additional_stopwords + tl_stopwords + collection_words

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stopwords])
df["Tweet"] = df["Tweet"].apply(lambda text: remove_stopwords(text))
df.head()

def remove_hashmentions(text):
    clean_tweet = re.sub('(@[a-z0-9]+)\w+',' ', text)
    clean_tweet = re.sub("#[A-Za-z0-9_]+"," ", clean_tweet)
    return clean_tweet
df['Tweet'] = df['Tweet'].apply(lambda x: remove_hashmentions(x))

english_punctuations = string.punctuation
punctuations_list = english_punctuations
def cleaning_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)
df['Tweet']= df['Tweet'].apply(lambda x: cleaning_punctuations(x))
df

def remove_repeating_chars(text):
    return re.sub(r'(.)1+', r'1', text)
df['Tweet'] = df['Tweet'].apply(lambda x: remove_repeating_chars(x))
df

def remove_URLs(text):
    return re.sub('((www.[^s]+)|(https?://[^s]+))','',text)
df['Tweet'] = df['Tweet'].apply(lambda x: remove_URLs(x))
df

def remove_numbers(text):
    return re.sub('[0-9]+', '', text)
df['Tweet'] = df['Tweet'].apply(lambda x: remove_numbers(x))
df["Tweet"].head()

import nltk
st = nltk.PorterStemmer()
def stemming_on_text(data):
    text = [st.stem(word) for word in data]
    return data
df['Tweet']= df['Tweet'].apply(lambda x: stemming_on_text(x))
df

lm = nltk.WordNetLemmatizer()
def lemmatizer_on_text(data):
    text = [lm.lemmatize(word) for word in data]
    return data
df['Tweet'] = df['Tweet'].apply(lambda x: lemmatizer_on_text(x))
df

"""MASKING WORDS"""

df["Tweet"] = df["Tweet"].str.replace("duque","")
df["Tweet"] = df["Tweet"].str.replace("yang","")
df["Tweet"] = df["Tweet"].str.replace("harry","")
df["Tweet"] = df["Tweet"].str.replace("roque","")
df["Tweet"] = df["Tweet"].str.replace("michael","")
df["Tweet"] = df["Tweet"].str.replace("duterte","")
df["Tweet"] = df["Tweet"].str.replace("lao","")
df["Tweet"] = df["Tweet"].str.replace("ong","")
df["Tweet"] = df["Tweet"].str.replace("du","")
df["Tweet"] = df["Tweet"].str.replace("galvez","")
df["Tweet"] = df["Tweet"].str.replace("leni","")
df["Tweet"] = df["Tweet"].str.replace("francisco","")
df["Tweet"] = df["Tweet"].str.replace("pharmally","")
df["Tweet"] = df["Tweet"].str.replace("diokno","")
df["Tweet"] = df["Tweet"].str.replace("chel","")
df["Tweet"] = df["Tweet"].str.replace("bbm","")
df["Tweet"] = df["Tweet"].str.replace("padilla","")
df["Tweet"] = df["Tweet"].str.replace("robin","")
df["Tweet"] = df["Tweet"].str.replace("kiko","")
df["Tweet"] = df["Tweet"].str.replace("syndicated","")
df["Tweet"] = df["Tweet"].str.replace("villar","")

df.value_counts(df["category_id"])

"""**GPU**"""

# %tensorflow_version 2.x
# import tensorflow as tf
# device_name = tf.test.gpu_device_name()
# #device_name = '/TPU:0'
# if device_name != '/device:GPU:0':
# #if device_name != '/TPU:0':
#   raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))

"""**TPU**"""

import tensorflow as tf

try:
  tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
  print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
except ValueError:
  raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

batch_size=32 * tpu_strategy.num_replicas_in_sync
print('Batch size:', batch_size)
AUTOTUNE = tf.data.experimental.AUTOTUNE

X = df.Tweet
y = df.category_id

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=0)

MAX_LEN = 280
MODEL_NAME = "jcblaise/roberta-tagalog-small"

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
    with tpu_strategy.scope():
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

tokenizer = RobertaTokenizer.from_pretrained("jcblaise/roberta-tagalog-small") #Tokenizer
X_train_tk = roberta_encode(X_train, tokenizer)
X_test_tk = roberta_encode(X_test, tokenizer)

early_stopping = EarlyStopping(monitor='loss', mode='min', patience=5)
with tpu_strategy.scope():
  model = build_model(3)
  history = model.fit(X_train_tk,y_train,epochs=50, batch_size=32, validation_data=(X_test_tk, y_test), validation_split=0.5, verbose=1, callbacks=[early_stopping])
  predicted = model.predict(X_test_tk)
  y_prediction = np.argmax (predicted, axis = 1)
  print(classification_report(y_test,y_prediction))  
  print(accuracy_score(y_test, y_prediction))

texts = ["our [MASK] LGU is really good .. it never stops working to help those in need. thank you very much "]
text_features = roberta_encode(texts,tokenizer)
predictions = model.predict(text_features)
predictions = np.argmax (predictions, axis = 1)
for text, predicted in zip(texts, predictions):
  print('"{}"'.format(text))
  print("  - Predicted as: '{}'".format(id_to_category[predicted]))
  print("")

texts = ["2 years and nothing has changed! hay grabe ka [MASK] palpak ka talaga!! "]
text_features = roberta_encode(texts,tokenizer)
predictions = model.predict(text_features)
predictions = np.argmax (predictions, axis = 1)
for text, predicted in zip(texts, predictions):
  print('"{}"'.format(text))
  print("  - Predicted as: '{}'".format(id_to_category[predicted]))
  print("")

texts = ["Vaccines works kaya dapat pataasin pa booster dose para mamaintain yan feb to march kasi tumaas vaccination tataas uli case pag nagwane yung mga 1st and 2nd dose lang nabakunahan kaya dapat pataasin booster "]
text_features = roberta_encode(texts,tokenizer)
predictions = model.predict(text_features)
predictions = np.argmax (predictions, axis = 1)
for text, predicted in zip(texts, predictions):
  print('"{}"'.format(text))
  print("  - Predicted as: '{}'".format(id_to_category[predicted]))
  print("")

texts = ["THE BEST AND THE ONLY CHOICE FOR PRESIDENT IS VP [MASK]! "]
text_features = roberta_encode(texts,tokenizer)
predictions = model.predict(text_features)
predictions = np.argmax (predictions, axis = 1)
for text, predicted in zip(texts, predictions):
  print('"{}"'.format(text))
  print("  - Predicted as: '{}'".format(id_to_category[predicted]))
  print("")

texts = ["Do not lockdown, it's a good policy. It does not stop the work ..."]
text_features = roberta_encode(texts,tokenizer)
predictions = model.predict(text_features)
predictions = np.argmax (predictions, axis = 1)
for text, predicted in zip(texts, predictions):
  print('"{}"'.format(text))
  print("  - Predicted as: '{}'".format(id_to_category[predicted]))
  print("")

texts = ["So wala po talaga kwenta ginagawa ng DOH?"]
text_features = roberta_encode(texts,tokenizer)
predictions = model.predict(text_features)
predictions = np.argmax (predictions, axis = 1)
for text, predicted in zip(texts, predictions):
  print('"{}"'.format(text))
  print("  - Predicted as: '{}'".format(id_to_category[predicted]))
  print("")

model.save_weights('/content/roberta-fil-small.h5')