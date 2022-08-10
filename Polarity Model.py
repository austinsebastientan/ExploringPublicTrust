#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.corpus import stopwords
import string
import re
from wordcloud import WordCloud 
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud,STOPWORDS
from sklearn import metrics

import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


# In[2]:


df = pd.read_csv('translated_tagged_tweets.csv')


# In[3]:


df.dropna(subset = ["Combined"], inplace=True)


# In[4]:


df


# In[5]:


df.value_counts(df["Combined"])


# In[6]:


## remove filipino dialect tweets
df = df[df.Combined != "Filipino Dialects"]
df = df[df.Combined != "Irrelevant"]
df = df.reset_index()


# In[7]:


df.value_counts(df["Combined"])


# In[8]:


col = ["tweet_trans","Combined"]
df = df[col]


# In[9]:


df.columns


# In[10]:


df.columns = ['Tweet', 'Label']


# In[11]:


df


# In[12]:


df.drop_duplicates(subset= 'Tweet', inplace = True)
df


# In[13]:


df.loc[df["Label"] == "Institutional-Negative", "Label"] = "Negative"
df.loc[df["Label"] == "Institutional-Positive", "Label"] = "Positive"
df.loc[df["Label"] == "Behavioral-Negative", "Label"] = "Negative"
df.loc[df["Label"] == "Behavioral-Positive", "Label"] = "Positive"
df.loc[df["Label"] == "Operational-Negative", "Label"] = "Negative"
df.loc[df["Label"] == "Operational-Positive", "Label"] = "Positive"


# In[14]:


df['category_id'] = df['Label'].factorize()[0]
from io import StringIO
category_id_df = df[['Label', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Label']].values)


# In[15]:


df.value_counts(df["Label"])


# In[16]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
df.groupby('Label').Tweet.count().plot.bar(ylim=0)
plt.show()


# In[17]:


df["Tweet"] = df["Tweet"].str.lower()
df


# In[18]:


additional_stopwords = ['rt','vs','covid19nnchernobyl', 'httpstcob7qi0shk7g','httpstcoirnjm1xvmy', 'covid19','abscbnnews','httpstcojawhu','outbluerst','covidph', 'httpstcozxjgaxblya', '2248071','20745','69','#covid19ph','rohtak','jind','sirsa','sonipat','faridabad','indiafightscorona','kurukshetra','gurgaon','covid19india','haryana', 'httpstcotpuiehnaga', 'httpstcocxcybcjjzb', 'duquen', 'deybest','n','tabogo', 'deybesta','tutu', 'estafa','httpstcfzmqvjhd' ]
collection_words = ['#covidph','#covid19ph','#coronavirus', '#covidph','#DOH', '#coronaph','#qcprotektodo','#gcq','resbakuna', 'social distancing','delta variant','lockdown','bakuna', '#Halalan2022', '#LeniRobredoForPresident']
tl_stopwords = ["akin","aking","ako","alin","am","amin","aming","ang","ano","anumang","apat","at","atin","ating","ay","bababa","bago","bakit","bawat","bilang","dahil","dalawa","dapat","din","dito","doon","gagawin","gayunman","ginagawa","ginawa","ginawang","gumawa","gusto","habang","hanggang","hindi","huwag","iba","ibaba","ibabaw","ibig","ikaw","ilagay","ilalim","ilan","inyong","isa","isang","itaas","ito","iyo","iyon","iyong","ka","kahit","kailangan","kailanman","kami","kanila","kanilang","kanino","kanya","kanyang","kapag","kapwa","karamihan","katiyakan","katulad","kaya","kaysa","ko","kong","kulang","kumuha","kung","laban","lahat","lamang","likod","lima","maaari","maaaring","maging","mahusay","makita","marami","marapat","masyado","may","mayroon","mga","minsan","mismo","mula","muli","na","nabanggit","naging","nagkaroon","nais","nakita","namin","napaka","narito","nasaan","ng","ngayon","ni","nila","nilang","nito","niya","niyang","noon","o","pa","paano","pababa","paggawa","pagitan","pagkakaroon","pagkatapos","palabas","pamamagitan","panahon","pangalawa","para","paraan","pareho","pataas","pero","pumunta","pumupunta","sa","saan","sabi","sabihin","sarili","sila","sino","siya","tatlo","tayo","tulad","tungkol","una","walang", "po", "mas","pang","lang","si","kay","ba","mo","naman","di","ba","yung","nga","kayo","yan", "anona", "anuna", "tang", "buking", "mag"]


# In[19]:




stopwords = nltk.corpus.stopwords.words('english') + additional_stopwords + tl_stopwords + collection_words


# In[20]:


def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stopwords])
df["Tweet"] = df["Tweet"].apply(lambda text: remove_stopwords(text))
df.head()


# In[21]:


def remove_hashmentions(text):
    clean_tweet = re.sub('(@[a-z0-9]+)\w+',' ', text)
    clean_tweet = re.sub("#[A-Za-z0-9_]+"," ", clean_tweet)
    return clean_tweet
df['Tweet'] = df['Tweet'].apply(lambda x: remove_hashmentions(x))


# In[22]:


english_punctuations = string.punctuation
punctuations_list = english_punctuations
def cleaning_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)
df['Tweet']= df['Tweet'].apply(lambda x: cleaning_punctuations(x))
df


# In[23]:


def remove_repeating_chars(text):
    return re.sub(r'(.)1+', r'1', text)
df['Tweet'] = df['Tweet'].apply(lambda x: remove_repeating_chars(x))
df


# In[24]:


def remove_URLs(text):
    return re.sub('((www.[^s]+)|(https?://[^s]+))','',text)
df['Tweet'] = df['Tweet'].apply(lambda x: remove_URLs(x))
df


# In[25]:


def remove_numbers(text):
    return re.sub('[0-9]+', '', text)
df['Tweet'] = df['Tweet'].apply(lambda x: remove_numbers(x))
df


# In[26]:


import nltk
st = nltk.PorterStemmer()
def stemming_on_text(data):
    text = [st.stem(word) for word in data]
    return data
df['Tweet']= df['Tweet'].apply(lambda x: stemming_on_text(x))
df


# In[27]:


lm = nltk.WordNetLemmatizer()
def lemmatizer_on_text(data):
    text = [lm.lemmatize(word) for word in data]
    return data
df['Tweet'] = df['Tweet'].apply(lambda x: lemmatizer_on_text(x))
df


# In[28]:


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
df["Tweet"] = df["Tweet"].str.replace("robredo","")


# In[29]:


plt.figure(figsize=(40,25))

# Pos
subset = df[df.Label == "Positive"]
text = subset.Tweet.values
cloud_pos = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          collocations=False,
                          width=2500,
                          height=1800
                         ).generate(" ".join(text))

plt.subplot(2, 2, 1)
plt.axis('off')
plt.title("Positive",fontsize=40)
plt.imshow(cloud_pos)


# Neg
subset = df[df.Label == "Negative"]
text = subset.Tweet.values
cloud_neg = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          collocations=False,
                          width=2500,
                          height=1800
                         ).generate(" ".join(text))

plt.subplot(2, 2, 2)
plt.axis('off')
plt.title("Negative",fontsize=40)
plt.imshow(cloud_neg)






plt.savefig('cloud_categories.png')
plt.show()


# In[30]:




tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='utf-8', ngram_range=(1, 3), stop_words='english')

features = tfidf.fit_transform(df.Tweet).toarray()
labels = df.category_id

tfidf_df = tfidf.transform(df["Tweet"])
df['tfidf'] = list(tfidf.fit_transform(df['Tweet']).toarray())


tfidf_model = 'tf_idf.pk'
# pickle.dump(tfidf, open(filename, 'wb'))


# In[31]:


features


# In[32]:


labels


# ## AutoML

# In[33]:


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.40, random_state=0)


# In[34]:


feature_names = tfidf.get_feature_names()
tfidf_df_dense = tfidf_df.todense()
lst1 = tfidf_df_dense.tolist()
df_features =  pd.DataFrame(lst1, columns = feature_names)
df_features


# In[35]:


df = df.reset_index(drop=True)
df


# In[36]:


pycaret_df = df_features.copy()
pycaret_df = pd.concat([pycaret_df,df.category_id], axis=1)
# pycaret_df = pycaret_df.dropna()
pycaret_df


# In[37]:


from pycaret.classification import *
pyModel = setup(data = pycaret_df, target = "category_id", session_id=123, train_size=0.6, fix_imbalance=False, fold=5, use_gpu=True) 
best = compare_models()


# In[38]:


from pycaret.classification import *
pyModel = setup(data = pycaret_df, target = "category_id", session_id=123, train_size=0.6, fix_imbalance=False, fold=5, use_gpu=True) 
rfc = create_model('rf')


# In[39]:


finalize_model(rfc)


# In[40]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=-1, oob_score=False, random_state=123, verbose=0,
                       warm_start=False)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[41]:


print(metrics.classification_report(y_test, y_pred, 
                                    target_names=df['Label'].unique()))


# In[42]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.Label.values, yticklabels=category_id_df.Label.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# In[43]:


#when adding the phrase "for president" the model predicts it as negative instead.
#---> negative connotation for the term president

texts = ["THE BEST AND THE ONLY CHOICE FOR PRESIDENT IS VP [MASK]!", "2 years and nothing has changed! hay grabe ka [MASK] palpak ka talaga!!","Do not lockdown, it is not a good policy. It stops the work of people ...","I just really thank god we're all there fully vaccinated - plus my grandmother who covid. And Buti Naconvince Yun to be vaccinated in Early Roll out of vaccines. She was infected with the plumber that repaired the house we were so important to mask", "our [MASK] LGU is really good .. it never stops working to help those in need. thank you very much", "So wala po talaga kwenta ginagawa ng DOH?", "I don't think VP [MASK] has the character of a President to effectively run the country. With the people backing her, she will be just a puppet like ex-Pres. [MASK] serving the interest of those who gave her the thrown. Kawawa ang Pinas!" ]
text_features = tfidf.transform(texts)
predictions = model.predict(text_features)
for text, predicted in zip(texts, predictions):
  print('"{}"'.format(text))
  print("  - Predicted as: '{}'".format(id_to_category[predicted]))
  print("")


# In[44]:


# save the model 
polarity_model = 'polarity_model.sav'
# pickle.dump(model, open(filename, 'wb'))


# In[45]:


#load model from disk 

loaded_model = pickle.load(open(polarity_model, 'rb'))
loaded_tfidf = pickle.load(open(tfidf_model, 'rb'))


# In[46]:



texts = ["THE BEST AND THE ONLY CHOICE FOR PRESIDENT IS VP [MASK]!", "2 years and nothing has changed! hay grabe ka [MASK] palpak ka talaga!!","Do not lockdown, it is not a good policy. It stops the work of people ...","I just really thank god we're all there fully vaccinated - plus my grandmother who covid. And Buti Naconvince Yun to be vaccinated in Early Roll out of vaccines. She was infected with the plumber that repaired the house we were so important to mask", "our [MASK] LGU is really good .. it never stops working to help those in need. thank you very much", "So wala po talaga kwenta ginagawa ng DOH?", "I don't think VP [MASK] has the character of a President to effectively run the country. With the people backing her, she will be just a puppet like ex-Pres. [MASK] serving the interest of those who gave her the thrown. Kawawa ang Pinas!" ]
text_features = loaded_tfidf.transform(texts)
predictions = loaded_model.predict(text_features)
output ={0: 'Negative', 1: 'Positive'}
for text, predicted in zip(texts, predictions):
  print('"{}"'.format(text))
  print("  - Predicted as: '{}'".format(id_to_category[predicted]))
  print("")


# In[47]:


model.feature_importances_


# In[48]:




N = 25
indices = np.argsort(model.feature_importances_)
feature_names = np.array(tfidf.get_feature_names())[indices]
unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
trigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 3][:N]
# print("# '{}':".format(Product))
print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))
print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))
print("  . Top trigrams:\n       . {}".format('\n       . '.join(trigrams)))


# In[ ]:




