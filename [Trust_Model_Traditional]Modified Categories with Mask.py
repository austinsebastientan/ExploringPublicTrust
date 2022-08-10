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



#Trial 2 
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
# df = df.drop_duplicates(keep='first', subset=['Tweet'])
# df = df.reset_index()
# df = df.rename(columns={'index':'orig_index'})
df


# In[13]:


df.loc[df["Label"] == "Institutional-Negative", "Label"] = "Institutional"
df.loc[df["Label"] == "Institutional-Positive", "Label"] = "Institutional"
df.loc[df["Label"] == "Behavioral-Negative", "Label"] = "Behavioral"
df.loc[df["Label"] == "Behavioral-Positive", "Label"] = "Behavioral"
df.loc[df["Label"] == "Operational-Negative", "Label"] = "Operational"
df.loc[df["Label"] == "Operational-Positive", "Label"] = "Operational"


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
df["Tweet"].head()


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


# #### Masking Words

# In[28]:


# df["Tweet"] = df["Tweet"].str.replace("duque","[MASK]")
# df["Tweet"] = df["Tweet"].str.replace("yang","[MASK]")
# df["Tweet"] = df["Tweet"].str.replace("harry","[MASK]")
# df["Tweet"] = df["Tweet"].str.replace("roque","[MASK]")
# df["Tweet"] = df["Tweet"].str.replace("michael","[MASK]")
# df["Tweet"] = df["Tweet"].str.replace("duterte","[MASK]")


# In[29]:


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


# In[30]:


df


# In[31]:


plt.figure(figsize=(40,25))

# BN
subset = df[df.Label == "Behavioral"]
text = subset.Tweet.values
cloud_bn = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          collocations=False,
                          width=2500,
                          height=1800
                         ).generate(" ".join(text))

plt.subplot(2, 3, 1)
plt.axis('off')
plt.title("Behavioral",fontsize=40)
plt.imshow(cloud_bn)


# IN
subset = df[df.Label == "Institutional"]
text = subset.Tweet.values
cloud_in = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          collocations=False,
                          width=2500,
                          height=1800
                         ).generate(" ".join(text))

plt.subplot(2, 3, 3)
plt.axis('off')
plt.title("Institutional",fontsize=40)
plt.imshow(cloud_in)




# ON
subset = df[df.Label == "Operational"]
text = subset.Tweet.values
cloud_on = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          collocations=False,
                          width=2500,
                          height=1800
                         ).generate(" ".join(text))

plt.subplot(2, 3, 2)
plt.axis('off')
plt.title("Operational",fontsize=40)
plt.imshow(cloud_on)

# OP
plt.savefig('cloud_categories.png')
plt.show()


# In[32]:




tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='utf-8', ngram_range=(1, 3), stop_words='english')

features = tfidf.fit_transform(df.Tweet).toarray()
labels = df.category_id

tfidf_df = tfidf.transform(df["Tweet"])
df['tfidf'] = list(tfidf.fit_transform(df['Tweet']).toarray())


# In[33]:


from sklearn.feature_selection import chi2
import numpy as np


## terms most correlated to the categories
N = 25
for Product, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  trigrams = [v for v in feature_names if len(v.split(' ')) == 3]
  print("# '{}':".format(Product))
  print("  . Most associated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
  print("  . Most associated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))
  print("  . Most associated trigrams:\n       . {}".format('\n       . '.join(trigrams[-N:])))


# # Trial 1

# In[34]:




X_train, X_test, y_train, y_test = train_test_split(df['Tweet'], df['Label'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)



# In[35]:


print(clf.predict(count_vect.transform(["How has the government been unable to provide mass testing and contact tracing yet?"])))


# In[36]:


print(clf.predict(count_vect.transform(["Harry Roque palpak"])))


# In[37]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit


models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
cv = 5
# cv = ShuffleSplit(n_splits=5, test_size=0.4, random_state=0)
cv_df = pd.DataFrame(index=range(cv * len(models)))
# print(cv_df)
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=cv)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
print(cv_df)


# In[38]:


import seaborn as sns

sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()


# In[39]:


cv_df.groupby('model_name').accuracy.mean()


# In[40]:


from sklearn.model_selection import train_test_split

model = LinearSVC()

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.4, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[41]:


from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.Label.values, yticklabels=category_id_df.Label.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# In[42]:


from IPython.display import display

for predicted in category_id_df.category_id:
  for actual in category_id_df.category_id:
    if predicted != actual and conf_mat[actual, predicted] >= 1:
      print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual], id_to_category[predicted], conf_mat[actual, predicted]))
      display(df.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['Label', 'Tweet']])
      print('')


# In[43]:


model.fit(features, labels)


# In[44]:


from sklearn.feature_selection import chi2

N = 25
for Product, category_id in sorted(category_to_id.items()):
  indices = np.argsort(model.coef_[category_id])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
  bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
  trigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 3][:N]
  print("# '{}':".format(Product))
  print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))
  print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))
  print("  . Top trigrams:\n       . {}".format('\n       . '.join(trigrams)))


# In[45]:


category_to_id.items()


# In[46]:


#when adding the phrase "for president" the model predicts it as negative instead.
#---> negative connotation for the term president

texts = ["THE BEST AND THE ONLY CHOICE FOR PRESIDENT IS VP [MASK]!"]
text_features = tfidf.transform(texts)
predictions = model.predict(text_features)
for text, predicted in zip(texts, predictions):
  print('"{}"'.format(text))
  print("  - Predicted as: '{}'".format(id_to_category[predicted]))
  print("")


# In[47]:


texts = ["2 years and nothing has changed! hay grabe ka [MASK] palpak ka talaga!!"]
text_features = tfidf.transform(texts)
predictions = model.predict(text_features)
for text, predicted in zip(texts, predictions):
  print('"{}"'.format(text))
  print("  - Predicted as: '{}'".format(id_to_category[predicted]))
  print("")


# In[48]:


texts = ["Do not lockdown, it's a good policy. It does not stop the work of people ..."]
text_features = tfidf.transform(texts)
predictions = model.predict(text_features)
for text, predicted in zip(texts, predictions):
  print('"{}"'.format(text))
  print("  - Predicted as: '{}'".format(id_to_category[predicted]))
  print("")


# In[49]:


texts = ["I just really thank god we're all there fully vaccinated - plus my grandmother who covid. And Buti Naconvince Yun to be vaccinated in Early Roll out of vaccines. She was infected with the plumber that repaired the house we were so important to mask"]
text_features = tfidf.transform(texts)
predictions = model.predict(text_features)
for text, predicted in zip(texts, predictions):
  print('"{}"'.format(text))
  print("  - Predicted as: '{}'".format(id_to_category[predicted]))
  print("")


# In[50]:


#translated from: ang galing talaga nang ating QC LGU.. hindi tumitigil gumawa nang paraan para matulungan 
#ang mga nangangailangan. marami pong salamat
texts = ["our [MASK] LGU is really good .. it never stops working to help those in need. thank you very much"]
text_features = tfidf.transform(texts)
predictions = model.predict(text_features)
for text, predicted in zip(texts, predictions):
  print('"{}"'.format(text))
  print("  - Predicted as: '{}'".format(id_to_category[predicted]))
  print("")


# In[51]:


#So wala po talaga kwenta ginagawa ng DOH at si DUQUE? when Duque is included it becomes BN

texts = ["So wala po talaga kwenta ginagawa ng DOH?"]
text_features = tfidf.transform(texts)
predictions = model.predict(text_features)
for text, predicted in zip(texts, predictions):
  print('"{}"'.format(text))
  print("  - Predicted as: '{}'".format(id_to_category[predicted]))
  print("")


# In[52]:


from sklearn import metrics
print(metrics.classification_report(y_test, y_pred, 
                                    target_names=df['Label'].unique()))


# In[53]:


print(metrics.accuracy_score(y_test,y_pred))


# # Trial 2 (AutoML)

# In[54]:


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.40, random_state=0)


# In[55]:


feature_names = tfidf.get_feature_names()
tfidf_df_dense = tfidf_df.todense()
lst1 = tfidf_df_dense.tolist()
df_features =  pd.DataFrame(lst1, columns = feature_names)
df_features


# In[56]:


df = df.reset_index(drop=True)
df


# In[57]:


pycaret_df = df_features.copy()
pycaret_df = pd.concat([pycaret_df,df.category_id], axis=1)
# pycaret_df = pycaret_df.dropna()
pycaret_df


# In[58]:


from pycaret.classification import *
pyModel = setup(data = pycaret_df, target = "category_id", session_id=123, train_size=0.6, fix_imbalance=False, fold=5, use_gpu=True) 
best = compare_models()


# In[73]:


metrics = get_metrics()
metrics


# In[60]:


evaluate_model(best)


# In[61]:


plot_model(best, plot = 'feature')


# In[62]:


plot_model(best, plot = "class_report")


# In[63]:


finalize_model(best)


# In[64]:


from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                     criterion='gini', max_depth=None, max_features='auto',
                     max_leaf_nodes=None, max_samples=None,
                     min_impurity_decrease=0.0, min_impurity_split=None,
                     min_samples_leaf=1, min_samples_split=2,
                     min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
                     oob_score=False, random_state=123, verbose=0,
                     warm_start=False)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[65]:


texts = ["So wala po talaga kwenta ginagawa ng DOH?"]
text_features = tfidf.transform(texts)
predictions = model.predict(text_features)
for text, predicted in zip(texts, predictions):
    print('"{}"'.format(text))
    print("  - Predicted as: '{}'".format(id_to_category[predicted]))
    print("")


# In[66]:


#translated from: ang galing talaga nang ating QC LGU.. hindi tumitigil gumawa nang paraan para matulungan 
#ang mga nangangailangan. marami pong salamat
texts = ["our [MASK] LGU is really good .. it never stops working to help those in need. thank you very much"]
text_features = tfidf.transform(texts)
predictions = model.predict(text_features)
for text, predicted in zip(texts, predictions):
  print('"{}"'.format(text))
  print("  - Predicted as: '{}'".format(id_to_category[predicted]))
  print("")


# In[67]:


#when adding the phrase "for president" the model predicts it as negative instead.
#---> negative connotation for the term president

texts = ["THE BEST AND THE ONLY CHOICE FOR PRESIDENT IS VP [MASK]!"]
text_features = tfidf.transform(texts)
predictions = model.predict(text_features)
for text, predicted in zip(texts, predictions):
  print('"{}"'.format(text))
  print("  - Predicted as: '{}'".format(id_to_category[predicted]))
  print("")


# In[68]:


texts = ["2 years and nothing has changed! hay grabe ka [MASK] palpak ka talaga!!"]
text_features = tfidf.transform(texts)
predictions = model.predict(text_features)
for text, predicted in zip(texts, predictions):
  print('"{}"'.format(text))
  print("  - Predicted as: '{}'".format(id_to_category[predicted]))
  print("")


# In[69]:


texts = ["Do not lockdown, it's not a good policy. It stops the work of people ..."]
text_features = tfidf.transform(texts)
predictions = model.predict(text_features)
for text, predicted in zip(texts, predictions):
  print('"{}"'.format(text))
  print("  - Predicted as: '{}'".format(id_to_category[predicted]))
  print("")


# In[70]:


texts = ["I just really thank god we're all there fully vaccinated - plus my grandmother who covid. And Buti Naconvince Yun to be vaccinated in Early Roll out of vaccines. She was infected with the plumber that repaired the house we were so important to mask"]
text_features = tfidf.transform(texts)
predictions = model.predict(text_features)
for text, predicted in zip(texts, predictions):
  print('"{}"'.format(text))
  print("  - Predicted as: '{}'".format(id_to_category[predicted]))
  print("")


# In[ ]:





# In[ ]:




