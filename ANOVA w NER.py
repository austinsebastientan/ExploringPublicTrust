#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
from pprint import pprint
import pandas as pd

import re
import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')
from nltk.corpus import stopwords
import string

from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

#!python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

# pd.set_option('display.max_rows', None)


# In[2]:


english_punctuations = string.punctuation
punctuations_list = english_punctuations
st = nltk.PorterStemmer()
lm = nltk.WordNetLemmatizer()

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

def stemming_on_text(data):
    text = [st.stem(word) for word in data]
    return data
def lemmatizer_on_text(data):
    text = [lm.lemmatize(word) for word in data]
    return data


def preprocess(df): 
    
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
    #stem
    df['Tweet']= df['Tweet'].apply(lambda x: stemming_on_text(x))
    #lemm
    df['Tweet'] = df['Tweet'].apply(lambda x: lemmatizer_on_text(x))
    
    
    return df


# In[4]:


def tokenization(text):
    text = re.split('\W+', text)
    return text


# In[5]:


df = pd.read_csv('translated_tagged_tweets.csv')
df.dropna(subset = ["Combined"], inplace=True)
df = df[df.Combined != "Filipino Dialects"]
df = df[df.Combined != "Irrelevant"]
df = df.reset_index()


# In[6]:


col = ["tweet_trans","Combined"]
df = df[col]


# In[7]:


df.columns = ['Tweet', 'Label']
df


# In[8]:


df.drop_duplicates(subset= 'Tweet', inplace = True)
df = df.reset_index()
df.drop('index', inplace=True, axis=1)
df


# # Preprocess

# In[9]:


df = preprocess(df)


# In[10]:


#remove bias words
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
df["Tweet"] = df["Tweet"].str.replace("robredo","")
df["Tweet"] = df["Tweet"].str.replace("chel","")
df["Tweet"] = df["Tweet"].str.replace("bbm","")
df["Tweet"] = df["Tweet"].str.replace("padilla","")
df["Tweet"] = df["Tweet"].str.replace("robin","")
df["Tweet"] = df["Tweet"].str.replace("kiko","")
df["Tweet"] = df["Tweet"].str.replace("syndicated","")
df["Tweet"] = df["Tweet"].str.replace("villar","")


# In[11]:


df


# In[12]:


df.loc[df["Label"] == "Institutional-Negative", "Label"] = "Institutional"
df.loc[df["Label"] == "Institutional-Positive", "Label"] = "Institutional"
df.loc[df["Label"] == "Behavioral-Negative", "Label"] = "Behavioral"
df.loc[df["Label"] == "Behavioral-Positive", "Label"] = "Behavioral"
df.loc[df["Label"] == "Operational-Negative", "Label"] = "Operational"
df.loc[df["Label"] == "Operational-Positive", "Label"] = "Operational"


# # Get all possible NER Labels

# In[13]:


nlp.entity.labels


# In[14]:


columns = ["CARDINAL", "DATE", "EVENT", "FAC","GPE", "LANGUAGE", "LAW", "LOC","MONEY","NORP","ORDINAL", "ORG", "PERCENT", "PERSON", "PRODUCT", "QUANTITY", "TIME", "WORK_OF_ART"]


# # NER

# In[15]:


def ner_label_count(text):
    doc = nlp(text)
    labels = [x.label_ for x in doc.ents]
    return Counter(labels)
    
def ner(text):
    doc = nlp(text)
    return ([(X.text, X.label_) for X in doc.ents])


# # Generate Compiled Data

# In[16]:


df['ner'] = df['Tweet'].apply(lambda x: ner(x))
df['ner_label'] = df['Tweet'].apply(lambda x: ner_label_count(x))


# In[17]:


## Check outputs
df = df.reset_index()
df.drop('index', inplace=True, axis=1)
df


# In[18]:


for x in columns:
    counts = []
    for i in range(len(df)):
        if x in df["ner_label"][i].keys():
            counts.append(df["ner_label"][i][x])
        else:
            counts.append(0)
    df[x] = counts
            


# In[19]:


df


# In[20]:


# df.to_csv('NER_features.csv')


# In[21]:


significant_features = []
for x in columns: 
    f_val,p_val = f_oneway(df[x][df['Label'] == 'Institutional'],df[x][df['Label'] == 'Behavioral'], df[x][df['Label'] == 'Operational'])
    if p_val < 0.05:
        print("ANOVA Test for feature : {}".format(x))
        print("F-Value = {} and p-value = {}".format(f_val,p_val))
        print("")
        significant_features.append(x)


# In[22]:



for x in significant_features:
    print("Conducting post hoc test for feature: {}".format(x))
    print(pairwise_tukeyhsd(endog = df[x], groups = df['Label'], alpha = 0.05))
    print("")


# In[ ]:




