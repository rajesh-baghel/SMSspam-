#!/usr/bin/env python
# coding: utf-8

# In[2]:


import nltk


# In[3]:


#nltk.download_shell()


# In[1]:


messages=[line.rstrip() for line in open('C:/Users/hp/Downloads/spam detector-project/smsspamcollection/SMSSpamCollection')]


# In[5]:


len(messages)


# In[6]:


messages[50]


# In[7]:


for message_no,message in enumerate(messages[:10]):
    print(message_no,message)
    print('\n')


# In[8]:


messages[0]


# In[9]:


## This indicates that this is a tab separation.TSV(Tab separted values)


# In[10]:


import pandas as pd


# In[11]:


messages=pd.read_csv('C:/Users/hp/Downloads/spam detector-project/smsspamcollection/SMSSpamCollection', sep='\t',
                               names=['label','message'])


# In[12]:


messages.head()


# ## Exploratory Data Analysis

# In[13]:


messages.describe()


# In[14]:


## Using groupby to separate ham and spam


# In[15]:


messages.groupby('label').describe()


# ### create new column to find the length of text messages

# In[16]:


messages['length']=messages['message'].apply(len)


# In[17]:


messages.head()


# ## Data Visualization

# In[18]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:


messages['length'].plot(bins=150,kind='hist')


# In[20]:


messages.length.describe()


# In[21]:


messages[messages['length']==910]['message'].iloc[0]


# In[22]:


## As this clearly shows that the length is a distinguishing feature between ham and spam 


# In[23]:


messages.hist(column='length',by='label',bins=60,figsize=(12,6))


# In[24]:


## This clear indicates that spam messages tend to have more characters


# ## Text pre-processing

# In[25]:


## our data is in text format. for classification task we have to convert a corpus to a vector format.


# In[26]:


import string


# In[27]:


string.punctuation


# In[28]:


from nltk.corpus import stopwords
stopwords.words('english')


# In[29]:


def text_process(mess):
    """
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    return [mes for mes in nopunc.split() if mes.lower() not in stopwords.words('english')]


# In[30]:


messages.head()


# In[31]:


## Tokenizing to convert the normal text strings in to a list of tokens


# In[32]:


messages['message'].head(5).apply(text_process)


# In[33]:


# Noticed that we have a lot of shorthands in our sms text data such as U, Nah etc.so stemming isn't going to work so great for our dataset. we're going for vectorization
# and currently we have message as lists of tokens now we have to convert each messages into vector.
# so scikit learn algorithm models can work.


# ## Vectorization

# In[34]:


# we doing it in three steps using bag-of-words model:
#1. count how many times does a word occur in each message(Term frequency)
#2. weigh the counts, so that frequent tokens get lower weight(inverse document frequency)
#3. Normalize the vectors to unit length


# In[35]:


# using countvectorizer to convert a collection of text documents to a matrix of token counts


# In[36]:


from sklearn.feature_extraction.text import CountVectorizer


# In[37]:


bow_transformer=CountVectorizer(analyzer=text_process).fit(messages['message'])


# In[38]:


print(len(bow_transformer.vocabulary_))


# In[39]:


#example text message and get its bag-of-words counts as a vector 


# In[40]:


mess4=messages['message'][3]


# In[41]:


print(mess4)


# In[42]:


bow4=bow_transformer.transform([mess4])


# In[43]:


print(bow4)


# In[44]:


# This means that there are seven unique words in mess4.
# Two of them appear twice.


# In[45]:


print(bow_transformer.get_feature_names()[4068])
print(bow_transformer.get_feature_names()[9554])


# In[46]:


#. Now we use .transform on bag-of-ords(bow).


# In[47]:


messages_bow=bow_transformer.transform(messages['message'])


# In[48]:


print('shape of sparse matrix: ', messages_bow.shape)
print('Amount of non-zero occurance: ', messages_bow.nnz)


# In[49]:


### After the counting, the weighting and normalization done with TF-IDF, using scikit-learn
###TF-IDF(term frequency -inverse document frequency)


# In[50]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[51]:


tfidf_transformer = TfidfTransformer().fit(messages_bow)


# In[52]:


#Transform the entire bag-of-words corpus


# In[53]:


messages_tfidf = tfidf_transformer.transform(messages_bow)


# In[54]:


print(messages_tfidf.shape)


# ## Training Model

# In[55]:


# using Naive Bayes Classifier Algorithm


# In[56]:


from sklearn.naive_bayes import MultinomialNB


# In[59]:


spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])


# In[60]:


all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)


# ## classification report

# In[61]:


from sklearn.metrics import classification_report


# In[82]:


print(classification_report(messages['label'],all_predictions))


# ## Train Test Split

# In[63]:


from sklearn.model_selection import train_test_split 


# In[72]:


from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.2)

print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))


# ## Creating a Data pipeline

# In[78]:


from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


# In[79]:


pipeline.fit(msg_train,label_train)


# In[80]:


predictions=pipeline.predict(msg_test)


# In[81]:


print(classification_report(predictions,label_test))


# In[ ]:




