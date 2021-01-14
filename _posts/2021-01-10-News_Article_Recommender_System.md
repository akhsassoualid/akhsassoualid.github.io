---
title: "A Recommender System for news article headlines"
excerpt: "Machine Learning, Recommender System, Word2vec, LDA, NMF, Euclidean"
mathjax: "true"
---
A recommender system is one of the most used tools by big tech companies on their website platforms. It helps to present more suitable suggestions for products, items, articles, news, to each client.
The algorithms that manage the recommender system can be classified to:


*   Non-personalized systems: It suggests the most trending items for a population based on a certain statistical criterion. it's like proposing the best seller book, or the best hotel in a town.
*   Collaborative filtering: It presents multiple suggestions of items based on data that concerns a population. In other words, it's driven by the statement **" tell me what's popular among my neighbors because I might like it too if it's similar to my preferences".**

*   Content-based filtering: it's an algorithm that suggests to the client more preposition based on his own data, like his rating, or choice of a movie, his review of an article. We can resume it in the following statement: **"show me more of the same of what I have liked, see, purchase before".** 

In this project, we will try to build a recommender system for an ensemble of articles of news. We have news headlines of many years. To get a quick result, we will get a sample, it concerns news article headlines for the year 2018.

**Our main goal is to use content-based filtering to build a recommender system that suggests more articles similar to what you have already read based on similar present words in each headline.**


```python
!pip install gdown
```
    


```python
# Libraries to import dataset, for plots 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from datetime import datetime

# pip library installation
!pip install unidecode
!pip install nltk

# Import text preprocessing libraries
import string
import spacy
import re
import unidecode
import nltk
from collections import Counter
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Other downloading
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')
nltk.download('wordnet')

# libraries to build the recommender system
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import  NMF, LatentDirichletAllocation
from gensim.models import Word2Vec
from sklearn.pipeline import make_pipeline
from sklearn.metrics import pairwise_distances
```
    

# **1.   Import, clean, and discover the dataset:**

# **1.1. Import dataset and applying some cleaning:**

We will import a dataset of 200K news headlines that start from the year 2012 to 2018, collected from [HuffoPost](https://www.huffpost.com/). The headlines concern different field such Politics, Sport, ....

It's saved in a json file that will be imported and build with it a recommender system for headlines. We add a column of the headline's year, delete any duplicates, and search for missing headlines.

The data can be found on [my link drive](https://drive.google.com/file/d/1mHDG0yMs9leCjHcr6h8FX5V10EpUqBIU/view?usp=sharing), or from the its [kaggle link](https://www.kaggle.com/rmisra/news-category-dataset).


```python
!gdown --id 1mHDG0yMs9leCjHcr6h8FX5V10EpUqBIU
```

    Downloading...
    From: https://drive.google.com/uc?id=1mHDG0yMs9leCjHcr6h8FX5V10EpUqBIU
    To: /content/News_Category_Dataset_v2.json
    83.9MB [00:00, 103MB/s] 
    


```python
# Import the json dataset
data = pd.read_json('News_Category_Dataset_v2.json', lines=True)
data['date'] = pd.to_datetime(data.date)
data['year'] = [x.year for x in data['date']]
```


```python
# Let's verify duplicates existance
duplica = data[data.duplicated('headline')==True]
data = data[data.duplicated('headline')==False]
print("the number of duplicated headlines is :{}".format(duplica.shape[0]))
```

    the number of duplicated headlines is :1509
    


```python
# checking missing values
data.isna().sum()
```




    category             0
    headline             0
    authors              0
    link                 0
    short_description    0
    date                 0
    year                 0
    dtype: int64



**Conclusions:**

---

It seems that we have 1509 duplicated headlines that will be deleted. Fortunately, we have no missing headlines.

# **1.2. Distribution plots of news articles headlines:**

We will figure out the distribution of all headlines by categories besides of the year of publication.


```python
# Discover the news by their category
fig, ax1 = plt.subplots(nrows=1, ncols=1,figsize=(13,3))
sns.set(style='darkgrid')
sns.countplot(x = 'category', data=data,order=data['category'].value_counts().index, ax=ax1)
plt.title('Distribution of news by catgeories')
plt.xticks(rotation=90)
plt.show()
```


![png](/img/blog/HeadlineRecommender/output_12_0.png)


**Conclusions:**

---
The distribution proves that most news articles are related to :

*   Politics
*   Entertainment
*   Wellness
*   Style and Beauty
*   Travel

The plot confirms too, that news articles are less interested in serious topics such as Education, Science, Tech, Environment.

 In order to have a clear idea about most important new's topics, we will plot the distribution of topics of 2016, 2017, and 2018.


```python
# Distribution of articles news per year
def count_plot_by(x, data, year):
  """ x : label of the column which is headline.
      data : the whole data.
      year : The chosen year of the news headlines"""
  n_data = data[data['year']==year]
  fig, ax = plt.subplots(figsize=(13,3))
  sns.set(style='darkgrid')
  sns.countplot(x=x, data = n_data, order=n_data[x].value_counts().index, ax=ax)
  plt.title('Distribution of news catgeory on '+str(year))
  plt.xticks(rotation=90)
  plt.show()

# Example of plots of 2016,2017, and 2018
count_plot_by(x='category', data=data, year=2015)
count_plot_by(x='category', data=data, year=2016)
count_plot_by(x='category', data=data, year=2017)
count_plot_by(x='category', data=data, year=2018)
```


![png](/img/blog/HeadlineRecommender/output_14_0.png)



![png](/img/blog/HeadlineRecommender/output_14_1.png)



![png](/img/blog/HeadlineRecommender/output_14_2.png)



![png](/img/blog/HeadlineRecommender/output_14_3.png)


 **Comment:**

---

The distribution per year of news articles shows that is the same. In fact, we can say that those articles have the same preoccupations and aren't changing.

# **2. Basic descriptive statistics of the sampled news artciles:**

In this part of the project, we will use a sample of the dataset. Since we have 179.335 articles, the sample will concern only 2018's news articles, so our new sampled data will contain 8494 articles.




```python
# Subset of data for the years of 2018
data = data[data['year'] >= 2018]
data = data.reset_index(drop=True)
```


```python
# Cheking some discreptive statistics of news dataset
print("The numbre of news categories is equal to: {}".format(len(pd.unique(data['category']))))
print("The number of authors is equal to:{}".format(len(pd.unique(data['authors']))))
print("The number of news we have is equal to:{}".format(data.shape[0]))
```

    The numbre of news categories is equal to: 26
    The number of authors is equal to:916
    The number of news we have is equal to:8547
    

**Conclusions:**

---

About the year of 2018, the data shows that:
* Headlines were distributed in 26 categories.
* 916 authors wrote those articles.
* 8547 articles were written.


At this stage of the project, we just want to have an idea about the distribution of words per each news article headline, So we added a new column that displays the number of words for each headline. It resembles, in fact, to a Gaussian distribution. Let's have a look at the distribution plot:


```python
# The distribution of descriptions 
data['headline_lenght'] = [len(data.iloc[i,1]) for i in range(data.shape[0])]
fig, ax2 = plt.subplots(nrows=1, ncols=1,figsize=(10,5))
sns.distplot(a = data['headline_lenght'], hist=True, color = 'darkblue', 
             hist_kws={'edgecolor':'black'}, ax=ax2, kde_kws={'linewidth': 4})
plt.title('Distribution of headlines lenght')
plt.xlabel("Headline's lenght")
plt.show()
```

    /usr/local/lib/python3.6/dist-packages/seaborn/distributions.py:2551: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    


![png](/img/blog/HeadlineRecommender/output_21_1.png)


# **3. Preprocess the nes artciles headlines:**

# **3.1. Preprocess the data with nltk:**

The previous descriptive step was necessary to get some insights about the sampled dataset.
The goal right now is to preprocess the data and make it suitable for the modeling part. We will:

*   Remove stopwords
*   Lower the headlines
*   Remove punctuations
*   Lemmatize the headlines
*   Remove URLs
*   Delete rare words

We build general_process function. It's composed of two subfunctions: 
*   **preprocess_text**: It takes each text and applies multiples cleaning steps to it.
*   **remove_rare_words**: After processing each headline at a time, we resemble all headlines in new data, Then "remove_rare_words" take the whole of them and remove rare words.

**Once the headlines are preprocessed, it will be saved in a new column 'headline_'**





```python
def general_process(data, column_headline, new_column_headline):
  """data : the original data that gather all information about each headline.
     column_headline : the column'name that contains the headlines in data.
     new_column_headline : the new column's name that will be added to data, and will be the preprocessed headlines.
  """
  # Define function to process each text alone:
  def preprocess_text(text):
    """text : the sentance or phrase or headline we want to process """
    stop_words = stopwords.words('english')
    lem = WordNetLemmatizer()
    # Lower Casing
    new_text = text.lower()

    # Remove stopwords and punctuations
    word_text = word_tokenize(new_text)
    punct = string.punctuation
    word_text = [word for word in word_text if not word in stop_words]
    word_text = [word for word in word_text if word.isalpha()]
    word_text = [word for word in word_text if len(word)>=2]
    word_text = [word.translate(str.maketrans('', '', punct)) for word in word_text]

    # Lemmatize the text data
    word_text = [lem.lemmatize(word) for word in word_text]
    text = ' '.join(word_text)
  
    # remove URl from text
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub(r'', text)
    return text

  # Define a function to remove rare words:
  def remove_rare_words(texts, n=30):
    """text : the sentance or phrase or headline we want to process.
       n = The number of rare words we want to remove, the default value is 30.
    """
    cnt = Counter()
    for text in texts:
      for word in text.split():
        cnt[word] +=1
    rare = [w for (w, wc) in cnt.most_common()[:-n-1:-1]]
    for text in texts:
      word_text = word_tokenize(text)
      word_text = [word for word in word_text if not word in rare]
      text = ' '.join(word_text)
    print("The most rare words are : {}".format(rare))
    return texts

  # Apply pre defined functions
  data = data[data[column_headline].apply(lambda x:len(x.split())>5)]
  data[new_column_headline] = data[column_headline].apply(preprocess_text)
  data[new_column_headline] = remove_rare_words(data[new_column_headline], 10)
  return data
```


```python
new_data = general_process(data=data, column_headline='headline', new_column_headline='headline_')
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:50: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    

    The most rare words are : ['heroine', 'recreational', 'ren', 'ape', 'menounos', 'squirrelly', 'cookie', 'furry', 'crazed', 'evacuation']
    

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:51: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    

**Conclusion:**

---
*  The function selected the following words as rare words: **'heroine', 'recreational', 'ren', 'ape', 'menounos', 'squirrelly', 'cookie', 'furry', 'crazed', 'evacuation'**.

*  We set the function to select only 10 rare words. 

**Comment:**

---
*  We exploit the previous function "general_process", and apply it to the data. We display some rows of data to have a look on the new_data.


```python
new_data.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>headline</th>
      <th>authors</th>
      <th>link</th>
      <th>short_description</th>
      <th>date</th>
      <th>year</th>
      <th>headline_</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CRIME</td>
      <td>There Were 2 Mass Shootings In Texas Last Week...</td>
      <td>Melissa Jeltsen</td>
      <td>https://www.huffingtonpost.com/entry/texas-ama...</td>
      <td>She left her husband. He killed their children...</td>
      <td>2018-05-26</td>
      <td>2018</td>
      <td>mass shooting texas last week tv</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ENTERTAINMENT</td>
      <td>Will Smith Joins Diplo And Nicky Jam For The 2...</td>
      <td>Andy McDonald</td>
      <td>https://www.huffingtonpost.com/entry/will-smit...</td>
      <td>Of course it has a song.</td>
      <td>2018-05-26</td>
      <td>2018</td>
      <td>smith join diplo nicky jam world cup official ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ENTERTAINMENT</td>
      <td>Hugh Grant Marries For The First Time At Age 57</td>
      <td>Ron Dicker</td>
      <td>https://www.huffingtonpost.com/entry/hugh-gran...</td>
      <td>The actor and his longtime girlfriend Anna Ebe...</td>
      <td>2018-05-26</td>
      <td>2018</td>
      <td>hugh grant marries first time age</td>
    </tr>
  </tbody>
</table>
</div>



# **4. Build a Recommender Engine for headlines:**

# **4.1. Explain the functionality of the "recommender_engine":**
We have prepared the recommender_engine function that will suggest for any given article or headline from the data its similar articles.

In fact, this similarity will be founded on Tf-idf embedding, and we will use two approaches in order to get the best word vectors representation of the headlines dataset.

LDA and NMF are the two main algorithms used in this step. They allow the decomposition of a matrix into two other matrices with specific dimensions.

However, the word2vec neural network model can be very useful to build headlines numerical representation. The word2vec uses the optimized weights of the neural network, **as a vector representation of each word**. The headline vector representation will be the sum of its words vector divided by the number of words composing the headline.

The recommender_engine function needs to be fed with some parameters:

*  data : the original data that gather all information about each headline.   
*  headline_article : Article's headline to look for its similar headlines.
*  n_similar_article : Number of headlines to suggest.
*  model : The model to use for embedding, LDA, NMF, or word2vec. 

**To note that we set a parameter "nc" to 25. It's the number of words vector's components. That means each word will be embedded to a vector of 25 elements.**

**To note also, that we use the Euclidean distance to compute the similarity between those wors vector representation.**


```python
def recommender_engine(data, headline_article, n_similar_article, model):
  """ data : the original data that contains the processed headlines.
      headline_article : Article's headline to look for its similar headlines.
      n_similar_article : Number of headlines to suggest.
      model : The model to use for embedding, LDA, NMF, or word2vec. 
  """
  # Set parameters
  nc = 25
  target_ix = data[data['headline']==headline_article].index.tolist()[0]
  proc_headlines = data['headline_']
  sentence_ix = data.index.tolist() # index each headline
  
  tfidf = TfidfVectorizer()
  csr_mat = tfidf.fit_transform(proc_headlines)
  # Word embedding with Tfidf using NMF decomposition 
  if model == 'nmf':
    nmf = NMF(n_components = nc)
    features = nmf.fit_transform(csr_mat)

  # Word embedding with Tfidf using LDA decomposition
  elif model == 'lda':
    lda = LatentDirichletAllocation(n_components=nc)
    features = lda.fit_transform(csr_mat)
  
  # Word embedding using word2vec
  elif model == "word2vec":
    # We fit the model to get the weights of each word
    headline_token = [[word for word in word_tokenize(headline)] for headline in proc_headlines]
    w2v = Word2Vec(sentences=headline_token, size = nc, min_count=1, workers=3)
    vocabulary = list(w2v.wv.vocab.keys())

    # we compute the weights for each headline
    w2v_headline = []
    for headline in proc_headlines:
      w2v_word = np.zeros(nc, dtype='float32')
      for word in headline.split():
        if word in vocabulary:
          w2v_word = np.add(w2v_word, w2v[word])
      w2v_word = np.divide(w2v_word, len(headline.split()))
      w2v_headline.append(w2v_word)
      features = w2v_headline

  df_features = pd.DataFrame(features, index = sentence_ix, columns = ["weights"+str(k) for k in range(nc)]) 
  target = np.array(df_features.iloc[target_ix,:]).reshape(1,-1)
  similarity = pairwise_distances(features, target, metric = "euclidean")

  df = pd.DataFrame({'category':data['category'],
                     'published':data['date'],
                     'headline': data['headline'],
                     'similarity':similarity.ravel().tolist()})
  # Show recommendations
  recom = df.nsmallest(n=n_similar_article, columns='similarity')
  print("*"*20, 'Target article', "*"*50)
  print(" "*3 + str(df['category'][target_ix])+' :', df['headline'][target_ix], " "*40)
  print("*"*20, 'Recommended articles', "*"*44)
  return recom.iloc[1:,]
```

# **4.2. Testing the recommender system based on articles profile:**

We will use the headlines of each news article to build a profile for it. By a profile, we mean a vector with a specific number of features that define the news article, in that case, it's 25 features or components.

The purpose is to assure that the function works well with different settings of its parameters.

Consequently, we try to test this function on a specific headline that is related to ex-president Trump. We talk about the article with index 2785 that states: "Woman Fired After Flipping Off Trump's Motorcade Sues Former Employer". Let's look for similar articles to this one.

# **4.2.1. Testing the recommender engine with NMF approach:**

*   We will ask the function to return the most 7 similar articles to the specified one.
*   We use 25 components for NMF factorization.


```python
recommender_engine(data=new_data, headline_article="Woman Fired After Flipping Off Trump's Motorcade Sues Former Employer",
                   n_similar_article=7, model="nmf")
```

    ******************** Target article **************************************************
       POLITICS : Woman Fired After Flipping Off Trump's Motorcade Sues Former Employer                                         
    ******************** Recommended articles ********************************************
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>published</th>
      <th>headline</th>
      <th>similarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5435</th>
      <td>POLITICS</td>
      <td>2018-02-17</td>
      <td>Former Mexican President: Mass Shootings Are C...</td>
      <td>0.020152</td>
    </tr>
    <tr>
      <th>2496</th>
      <td>POLITICS</td>
      <td>2018-04-10</td>
      <td>Retired General Schools Ivanka Trump After She...</td>
      <td>0.020440</td>
    </tr>
    <tr>
      <th>5033</th>
      <td>POLITICS</td>
      <td>2018-02-23</td>
      <td>Trump: Armed Teachers Would Have 'Shot The Hel...</td>
      <td>0.020808</td>
    </tr>
    <tr>
      <th>7568</th>
      <td>POLITICS</td>
      <td>2018-01-15</td>
      <td>Haitians In Florida Protest Trump's 'Shithole'...</td>
      <td>0.021232</td>
    </tr>
    <tr>
      <th>5568</th>
      <td>POLITICS</td>
      <td>2018-02-15</td>
      <td>Trump's Budget Cuts Millions Of Dollars From G...</td>
      <td>0.021418</td>
    </tr>
    <tr>
      <th>7298</th>
      <td>ENTERTAINMENT</td>
      <td>2018-01-18</td>
      <td>'The Magic School Bus' Flies Inside Trump's Bo...</td>
      <td>0.021661</td>
    </tr>
  </tbody>
</table>
</div>



# **4.2.2. Testing the recommender engine with LDA approach:**

*   We use 25 components for the LDA decomposition, and the euclidean disatnce to compute similarity.


```python
recommender_engine(data=new_data, headline_article="Woman Fired After Flipping Off Trump's Motorcade Sues Former Employer",
                   n_similar_article=7, model="lda")
```

    ******************** Target article **************************************************
       POLITICS : Woman Fired After Flipping Off Trump's Motorcade Sues Former Employer                                         
    ******************** Recommended articles ********************************************
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>published</th>
      <th>headline</th>
      <th>similarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2500</th>
      <td>MEDIA</td>
      <td>2018-04-10</td>
      <td>Lou Dobbs Flips Out On Live TV, Urges Trump To...</td>
      <td>0.000263</td>
    </tr>
    <tr>
      <th>2226</th>
      <td>POLITICS</td>
      <td>2018-04-14</td>
      <td>Judge: Transgender People A Protected Class, A...</td>
      <td>0.000276</td>
    </tr>
    <tr>
      <th>794</th>
      <td>POLITICS</td>
      <td>2018-05-10</td>
      <td>Trump Is Going To Indiana To Slam Joe Donnelly...</td>
      <td>0.000587</td>
    </tr>
    <tr>
      <th>1240</th>
      <td>QUEER VOICES</td>
      <td>2018-05-02</td>
      <td>2 Texas Men Who Used Grindr To Assault, Rob Ga...</td>
      <td>0.001067</td>
    </tr>
    <tr>
      <th>1589</th>
      <td>POLITICS</td>
      <td>2018-04-26</td>
      <td>Trump Says He Did Stay Overnight In Moscow, Cl...</td>
      <td>0.001135</td>
    </tr>
    <tr>
      <th>413</th>
      <td>POLITICS</td>
      <td>2018-05-18</td>
      <td>Rudy Giuliani Reverses Trump Team's Position, ...</td>
      <td>0.001186</td>
    </tr>
    <tr>
      <th>1034</th>
      <td>POLITICS</td>
      <td>2018-05-07</td>
      <td>Trump Urges West Virginia Voters Not To Back D...</td>
      <td>0.001281</td>
    </tr>
  </tbody>
</table>
</div>



# **4.2.3. Testing the recommender engine with word2vec approach:**

*   We use 25 weights for the word2vec approach, and the euclidean disatnce to compute similarity.


```python
recommender_engine(data=new_data, headline_article="Woman Fired After Flipping Off Trump's Motorcade Sues Former Employer",
                   n_similar_article=7, model="word2vec")
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:38: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
    

    ******************** Target article **************************************************
       POLITICS : Woman Fired After Flipping Off Trump's Motorcade Sues Former Employer                                         
    ******************** Recommended articles ********************************************
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>published</th>
      <th>headline</th>
      <th>similarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2787</th>
      <td>POLITICS</td>
      <td>2018-04-04</td>
      <td>All The Ways Trump Amped Up His Immigration Cr...</td>
      <td>0.023817</td>
    </tr>
    <tr>
      <th>4447</th>
      <td>POLITICS</td>
      <td>2018-03-06</td>
      <td>George W. Bush Reportedly Sounds Off On Trump:...</td>
      <td>0.023872</td>
    </tr>
    <tr>
      <th>2158</th>
      <td>POLITICS</td>
      <td>2018-04-16</td>
      <td>James Comey Gives A 'Strange Answer' When Aske...</td>
      <td>0.024146</td>
    </tr>
    <tr>
      <th>8057</th>
      <td>MEDIA</td>
      <td>2018-01-08</td>
      <td>Lindsey Graham: Only One Person On Earth Will ...</td>
      <td>0.024240</td>
    </tr>
    <tr>
      <th>5394</th>
      <td>POLITICS</td>
      <td>2018-02-18</td>
      <td>Trump Cites Facebook Executive In 'Fake News' ...</td>
      <td>0.024274</td>
    </tr>
    <tr>
      <th>4143</th>
      <td>POLITICS</td>
      <td>2018-03-11</td>
      <td>Trump Courting Clinton Impeachment Lawyer Amid...</td>
      <td>0.024367</td>
    </tr>
  </tbody>
</table>
</div>



**Conclusion:**

---
In general, it seems that all algorithm worked well. We used an example of a news that is related to the word "trump" and we got different suggestions of news that are related too to the word "trump".

# **5. Count number of POS for each article**

With the following steps, we want to add more variables to the dataset. The purpose is to count the number of POS of each article. In that case, each the counting of each POS in an article will be considered as a variable.

We define a function "spacy_pos" that could define POS in a text and return their counting.To note that POS refers to Part Of Speech, which are the grammatically tags of each word.


```python
# Function that return the POS of a text
nlp = spacy.load('en_core_web_sm')
def spacy_pos(text, want=False):
  """text : A given headline
     want : If True the function return the number of each POS in the text.
            Else it returns only the POS of the text.
  """
  doc = nlp(text)
  if want == False:
    pos_tokens = [token.pos_ for token in doc]
    count = dict(Counter(pos_tokens))
    return count
  else:
    pos_tokens = [(token.text, token,pos_) for token in doc]
    return pos_tokens
```

We use the previous function to select specific unique POS for each headline


```python
# To get unique POS
import itertools
pos_list = [spacy_pos(text) for text in data['headline']]
pos_list = list(itertools.chain(*pos_list))
pos_list = np.unique(np.array(pos_list))
pos_list = list(pos_list)
```


```python
# List of unique POS in all the healdines
pos_list
```




    ['ADJ',
     'ADP',
     'ADV',
     'AUX',
     'CCONJ',
     'DET',
     'INTJ',
     'NOUN',
     'NUM',
     'PART',
     'PRON',
     'PROPN',
     'PUNCT',
     'SCONJ',
     'SPACE',
     'SYM',
     'VERB',
     'X']



Now, we can count start counting the number of each POS in a headline. Those counting will be added to the original data "data". They can be considered as new features.


```python
# Let's count number of POS in each headline
pos_data = pd.DataFrame(columns=pos_list)
i = 0
for text in data["headline"]:
  x = dict(Counter(spacy_pos(text)))
  for pos in pos_list:
    if pos not in x:
      pos_data.loc[i,pos] = 0
    else:
      pos_data.loc[i,pos] = x[pos]
  i+=1
```


```python
# Merge data of POS counting with the original data
pos_data = pd.concat([data, pos_data], axis=1)
pos_data.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>headline</th>
      <th>authors</th>
      <th>link</th>
      <th>short_description</th>
      <th>date</th>
      <th>year</th>
      <th>ADJ</th>
      <th>ADP</th>
      <th>ADV</th>
      <th>AUX</th>
      <th>CCONJ</th>
      <th>DET</th>
      <th>INTJ</th>
      <th>NOUN</th>
      <th>NUM</th>
      <th>PART</th>
      <th>PRON</th>
      <th>PROPN</th>
      <th>PUNCT</th>
      <th>SCONJ</th>
      <th>SPACE</th>
      <th>SYM</th>
      <th>VERB</th>
      <th>X</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CRIME</td>
      <td>There Were 2 Mass Shootings In Texas Last Week...</td>
      <td>Melissa Jeltsen</td>
      <td>https://www.huffingtonpost.com/entry/texas-ama...</td>
      <td>She left her husband. He killed their children...</td>
      <td>2018-05-26</td>
      <td>2018</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ENTERTAINMENT</td>
      <td>Will Smith Joins Diplo And Nicky Jam For The 2...</td>
      <td>Andy McDonald</td>
      <td>https://www.huffingtonpost.com/entry/will-smit...</td>
      <td>Of course it has a song.</td>
      <td>2018-05-26</td>
      <td>2018</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ENTERTAINMENT</td>
      <td>Hugh Grant Marries For The First Time At Age 57</td>
      <td>Ron Dicker</td>
      <td>https://www.huffingtonpost.com/entry/hugh-gran...</td>
      <td>The actor and his longtime girlfriend Anna Ebe...</td>
      <td>2018-05-26</td>
      <td>2018</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



# **6. Application Deployement :**
To have an end to end project, we would like to serve our model through a web application and make it available for others to test out.

We split the project into 3 python files: 
*  preprocessing.py : A file that process the headlines and prepare them for embedding.
*  recommender.py : A file that builds a numerical representation of each headline based on three algorithms.
*  app.py : A file in order to run the application on streamlit.

Note that the folder "datasets" contains the original dataset, and the preprocessed dataset that was saved from the notebook.

You can take a look and follow the steps on my [repository](https://github.com/akhsassoualid/Headline_Recommender) to run the streamlit application on your local machine.

In addition, a Dockerfile was added in purpose to containerize the application and make it easy to deploy.


# **7. Conclusion**

To build a recommender engine, two main steps are important:
*   The preprocessing of data should be executed properly as a preparation for the next step.

*   The second step is based on the embedding algorithms. Building numerical representation for each headline is essential to compute the similarity between them.

*   Note that more embedding algorithms can and will be added, we talk about BERT, GloVe, Fastext.

*   With the NMF, LDA, and word2vec embedding algorithms, we got good results.


**The Application can be improved to another level. It's the idea that we can suggest similar headlines, not only of those present in the dataset, but also for outdoor headlines. This step could allow the application a self learning approach that smooths the work of a data scientist.**

# **References:**
- About word2vec:
 - https://www.youtube.com/watch?v=pOqz6KuvLV8&t=489s&ab_channel=Omnology.
 - https://machinelearningmastery.com/develop-word-embeddings-python-gensim/

- About LDA : 
 - https://www.mygreatlearning.com/blog/understanding-latent-dirichlet-allocation/

- About NMF :  
 - https://medium.com/logicai/non-negative-matrix-factorization-for-recommendation-systems-985ca8d5c16c

- About Recommender system tutorial :  
  - https://www.kaggle.com/vikashrajluhaniwal/recommending-news-articles-based-on-read-articles
