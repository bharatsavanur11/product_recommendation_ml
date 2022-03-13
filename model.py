#!/usr/bin/env python
# coding: utf-8

# In[858]:



import pandas as pd
import numpy as np
from collections import defaultdict
from collections import Counter
import csv
import re 
import string
from datetime import datetime
import time
from imblearn.over_sampling import SMOTE

# Visual Libraries
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

#Standard Displa format
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_colwidth', 300)



'V'# NLTK libraries
import nltk
nltk.download('all')
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

# Modelling
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics.pairwise import cosine_similarity


# In[859]:


df = pd.read_csv('sample30.csv')


# # Exploratory Data Analysis

# In[860]:


# Shape
print("Shape :", df.shape)

# Columns
print("Columns :")
print(df.columns)


# In[861]:


#Other Info
print(df.info())


# In[862]:


# Finding the count of missing values in each columns.
print("Missing Value Count :")
print(df.isnull().sum())


# In[863]:


#Finding the percentage of missing values in each columns.
print("Percentage of missing values :")
print(df.isna().mean().round(4) * 100)


# In[864]:


# Drop the NULL value rows of reviews_text, reviews_title, reviews_username, user_sentiment, reviews_date, manufacture. 
df = df[df['reviews_text'].notna()]
df = df[df['reviews_title'].notna()]
df = df[df['reviews_username'].notna()]
df = df[df['user_sentiment'].notna()]
df = df[df['reviews_date'].notna()]
df = df[df['manufacturer'].notna()]


# In[865]:


#Correcting the values where  review_rating is 5 and  reviews as Negative as they should be Positive
df.loc[(df["user_sentiment"] == "Negative") & (df["reviews_rating"] == 5) , "user_sentiment"] = 'Positive'
df.loc[(df["user_sentiment"] == "Positive") & (df["reviews_rating"] == 1) , "user_sentiment"] = 'Negative'


# In[866]:


df.rename(columns={'reviews_username' : 'user_id','name':'product_name'}, inplace=True)


# In[867]:


# Shape of Dataset: Total of 14 columns are there now.
df.info()


# In[868]:


# Convert the user sentiment column into binary values: Positive to 1 and Negative to 0.

def get_sentiment_binary(x):
    if(x== 'Positive'):
        return 1
    else:
        return 0

#Convert user_sentiment string into binary.
df['user_sentiment']=df['user_sentiment'].apply(get_sentiment_binary)


# # Data Cleaning and Text Processing

# In[869]:


# Merging the data to get more words and context
df['Review'] = df['reviews_title'] + " " + df['reviews_text'] 


# In[870]:


df


# In[871]:


# Lowercasing the reviews and title column
df = df.astype({"Review": str}, errors='raise') 


# In[872]:


# Lower Casing the REview Columns
df['Review'] = df['Review'].apply(lambda x : x.lower())


# In[873]:


df


# In[874]:


# Remove punctuation 
df['Review'] = df['Review'].str.replace('[^\w\s]','')


# In[875]:


# Remove Stopwords
stop = stopwords.words('english')
df['Review'] = df['Review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# In[876]:


# Print the first review from row1
df['Review'].head()


# In[877]:


lemmatizer = nltk.stem.WordNetLemmatizer()
wordnet_lemmatizer = WordNetLemmatizer() 


# In[878]:


def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


# In[879]:


def lemmatize_sentence(sentence):
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)


# In[880]:


# Apply Lemmetisation
df['Review']=df['Review'].apply(lambda x: lemmatize_sentence(x))


# In[881]:


def scrub_words(text):
    """Basic cleaning of texts."""
    
    # remove html markup
    text=re.sub("(<.*?>)","",text)
    
    #remove non-ascii and digits
    text=re.sub("(\\W|\\d)"," ",text)
    
    #remove whitespace
    text=text.strip()
    return text


# In[882]:


df['Review']=df['Review'].apply(lambda x: scrub_words(x))


# In[883]:


df_1 = df.copy()


# # Defining Features in this Data Set`

# In[884]:


x=df['Review'] 
y=df['user_sentiment']


# In[885]:


# #Distribution of the target variable data in terms of proportions.
print("Percent of 1s: ", 100*pd.Series(y).value_counts()[1]/pd.Series(y).value_counts().sum(), "%")
print("Percent of 0s: ", 100*pd.Series(y).value_counts()[0]/pd.Series(y).value_counts().sum(), "%")


# # Creating TFIDF Vectorizer with train and test split

# In[886]:


seed = 50 

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed)


# In[887]:


word_vectorizer = TfidfVectorizer(
    strip_accents='unicode',    # Remove accents and perform other character normalization during the preprocessing step. 
    analyzer='word',            # Whether the feature should be made of word or character n-grams.
    token_pattern=r'\w{1,}',    # Regular expression denoting what constitutes a “token”, only used if analyzer == 'word'
    ngram_range=(1, 3),         # The lower and upper boundary of the range of n-values for different n-grams to be extracted
    stop_words='english',
    sublinear_tf=True)

word_vectorizer.fit(X_train)    # Fiting it on Train
train_word_features = word_vectorizer.transform(X_train)  # Transform on Train


# In[888]:


## transforming the train and test datasets
X_train_transformed = word_vectorizer.transform(X_train.tolist())
X_test_transformed = word_vectorizer.transform(X_test.tolist())

# # Print the shape of each dataset.
print('X_train_transformed', X_train_transformed.shape)
print('y_train', y_train.shape)
print('X_test_transformed', X_test_transformed.shape)
print('y_test', y_test.shape)


# # Model 1 With Simple Logistic Regression

# In[889]:


time1 = time.time()

logit = LogisticRegression()
logit.fit(X_train_transformed,y_train)

time_taken = time.time() - time1
print('Time Taken: {:.2f} seconds'.format(time_taken))


# In[890]:


# Prediction Train Data
y_pred_train= logit.predict(X_train_transformed)

#Model Performance on Train Dataset
print("Logistic Regression accuracy", accuracy_score(y_pred_train, y_train))
print(classification_report(y_pred_train, y_train))


# In[891]:


# Prediction Test Data
y_pred_test = logit.predict(X_test_transformed)

#Model Performance on Test Dataset
print("Logistic Regression accuracy", accuracy_score(y_pred_test, y_test))
print(classification_report(y_pred_test, y_test))


# In[892]:


# Create the confusion matrix for Random Forest.

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics


print("Confusion matrix for train and test set")

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)

# confusion matrix for train set
cm_train = metrics.confusion_matrix(y_train, y_pred_train)
sns.heatmap(cm_train/np.sum(cm_train), annot=True , fmt = ' .2%')
# help(metrics.confusion_matrix)

plt.subplot(1,2,2)

# confusion matrix for the test data
cm_test = metrics.confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm_test/np.sum(cm_test), annot=True , fmt = ' .2%')

plt.show()


# # Recommendation Engine

# In[930]:


# import libraties
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


# In[931]:


# Reading ratings file from GitHub. # MovieLens
ratings = pd.read_csv('sample30.csv' , encoding='latin-1')


# # Encoding the data so that it can be passed for training recommendation system

# In[932]:


#ratings_updated = pd.DataFrame({ratings['id'],ratings['reviews_rating']})
ratings_updated = ratings[['name','reviews_rating','reviews_username']]
ratings_updated = ratings_updated.rename(columns={"name":"id"})
ratings_updated


# # Get Label Encoder Mappings that can be used in later stages

# In[935]:


# Test and Train split of the dataset.
from sklearn.model_selection import train_test_split
train, test = train_test_split(ratings_updated, test_size=0.30, random_state=31)


# In[936]:


print(train.shape)
print(test.shape)


# In[937]:


ratings_pivot = train.pivot_table(index='reviews_username',
                             columns='id',
                             values='reviews_rating')
ratings_pivot = ratings_pivot.replace(np.nan,0)
ratings_pivot


# In[938]:


type(ratings_pivot)


# In[939]:


df = ratings_pivot.iloc[6974]
df


# In[940]:


ratings_pivot.shape


# In[941]:


train.shape


# In[942]:


dummy_train = train.copy()


# In[943]:


# The ratoings note rated by user is marked as 1 for prediction. 
dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x>=1 else 1)


# In[944]:


dummy_train = dummy_train.pivot_table(index='reviews_username',
                             columns='id',
                             values='reviews_rating',
                            fill_value=1)


# In[945]:


dummy_train.shape


# # Cosine Similarity

# In[946]:


from sklearn.metrics.pairwise import pairwise_distances
# creating user similarity matrix
user_corel = 1 - pairwise_distances(ratings_pivot,metric='cosine')
print(user_corel)


# In[947]:


user_corel.shape


# In[948]:


user_corel[user_corel<0]=0
user_corel


# In[949]:


user_predicted_ratings = np.dot(user_corel, ratings_pivot.fillna(0))
user_predicted_ratings


# # Using adjusted Cosine

# In[950]:


# Create a user-movie matrix.
df_pivot = train.pivot_table(index='reviews_username', columns='id',values='reviews_rating')


# In[951]:


mean = np.nanmean(df_pivot, axis=1)
df_subtracted = (df_pivot.T-mean).T


# In[952]:


from sklearn.metrics.pairwise import pairwise_distances


# In[953]:


# Creating the User Similarity Matrix using pairwise_distance function.
user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation)


# In[954]:


user_correlation.shape


# # Prediction

# In[955]:


user_predicted_ratings.shape


# In[956]:


user_final_rating = np.multiply(user_predicted_ratings,dummy_train)
user_final_rating.head()


# In[957]:


d = user_final_rating.loc['00dog3'].sort_values(ascending=False)[0:5]
d


# In[969]:


# save the respective files and models through Pickle 
import pickle
pickle.dump(logit,open('logit_model.pkl', 'wb'))
# loading pickle object
logit =  pickle.load(open('logit_model.pkl', 'rb'))

pickle.dump(word_vectorizer,open('word_vectorizer.pkl','wb'))
# loading pickle object
word_vectorizer = pickle.load(open('word_vectorizer.pkl','rb'))

# Load the pickle file that was meant 
user_final_rating =  pickle.load(open('user_final_rating.pkl', 'rb'))


# In[970]:


# Define a function to recommend top 5 filtered products to the user.
def recommendations(user_input):
    # Add the unuser check for now as we do not have reviewes for him
    # The recommendations would be random for that person.
    try:
        print(user_final_rating.loc[user_input])
    except:
        return ''

    if user_final_rating.loc[user_input] is not None:
        d = user_final_rating.loc[user_input].sort_values(ascending=False)[0:20]
        i = 0
        a = {}
        for prod_name in d.index.tolist():
            product_name = prod_name
            print(product_name)
            product_name_review_list = df_1[df_1['product_name'] == product_name]['Review'].tolist()
            print(type(product_name_review_list))
            if len(product_name_review_list) > 0:
                print("Should go into vectorized")
                features = word_vectorizer.transform(product_name_review_list)
                logit.predict(features)
                a[product_name] = logit.predict(features).mean() * 100
        print(a)
        b = pd.Series(a).sort_values(ascending=False).head(5).index.tolist()
        print(b)
        return b


# In[971]:


recommendations('joshua')

