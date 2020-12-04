# capstone_2


<p align="center">

  <img src="https://fivethirtyeight.com/wp-content/uploads/2017/10/twitterpolitics-4x3.jpg?w=900">

</p>

I proposed an examination of Tweets and Twitter data regarding the LA city council election for council district (CD) 4 between incumbent David Ryu and Nithya Raman.

This election was noteworthy for a number of reasons: 
  * the incumbent David Ryu lost
  * Raman was a political newcomer
  * Ryu already amassed $800,000 for his campaign when Raman announced
  * Raman's campaign was grassroots; entirely funded by small donations (no PAC money, no corporate money)
  * Raman very far left, Ryu centrist-moderate
  * COVID outbreak in middle of campaign
  * National attention (Bernie Sanders endorsed Nithya, Hillary/Pelosi/Feinstein endorsed Ryu)
  * Being LA, large celebrity endorsements on both sides

I propose using a combination of Twitter API, tweepy, getoldtweets3, python-twitter libraries to look at tweets possibly going back as far as August 7, 2019 when Nithya announced her campaign. Open to suggestions on how to get the Twitter data, will be experimenting this weekend before solo week.

I would like to focus on NLP, exploring the tweets using unsupervised learning (K-means), perform sentiment analysis, as well as hopefully use the geographic data. Additionally (time permitting) would like to try out document clustering (possibly with Non-negative Matrix Factorization).

Could platform priorities be detected or predicted by the tweets of the candidates? Do Raman or Ryu supporters have common traits? Can sentiment be detected beyond positive/negative (i.e. pro-Raman/anti-Ryu or vice-versa)?

<p align="center">

  <img src="https://techcrunch.com/wp-content/uploads/2015/10/twitter-politics.png?w=900">

</p>

## Scraping Data
I was hoping to get clean Twitter data and metadata about the tweets in JSON format directly from the Twitter API in order to begin my project. Unfortunately it seems that fairly recently Twitter locked down much of their API and thus access to the data I needed. I did sign up for a Twitter developer account (and by process had to describe my project to Twitter to get approved), however with a free account I could only access Tweets from the past rolling 7 day period. Not a big enough window to even go back to the date of the election, let alone analyze the past Twitter activity of the candidates.

In locking down the API, Twitter also apparently cleaned up some "loopholes" that were utilized by some of the most common Python twitter scraping libraries, rendering them mostly useless in their current state.

Instead, I ended up using Selenium (highly recommend for ease of use, effectiveness, and it's is really fun) to create a program that can automatically scroll and scrape the most pertinent data (Twitter handle, timestamp, text, # of likes, # of comments, and # of retweets). I also parsed out and saved any emojis that I hoped to use (and maybe will in the future). I then saved the data in .csv format.

## Cleaning Data
Because of the way I wrote the program (basically a list of lists), it was easy to use Pandas and read the .csv files into DataFrames.


<p align="center">

  <img src="/images/df_example.png?w=400">
	
</p>


# EDA

To start I just wanted to make some visuals of the data to see what I had.

Nithya           |  David
:-------------------------:|:-------------------------:
![](/images/nithya_tweet_count.png)  |  ![](/images/david_tweet_count.png)
:-------------------------:|:-------------------------:
![](/images/nithya_word_per_tweet.png)  |  ![](/images/david_word_per_tweet.png)

Here are the most common words per candidate:

# Word cloud

And here are the most common hashtags:

# Word cloud

## Classification and Feature Engineering

Checked on effectiveness of `CountVectorizer()` and `TfidfVectorizer` on getting all this text into a usable numerical form:

```python

count_vect = CountVectorizer(lowercase=True, tokenizer=None, stop_words='english',
                             analyzer='word', max_features=1000)

tfidf = TfidfVectorizer(lowercase=True, tokenizer=None, stop_words='english',
                             analyzer='word', max_features=1000)
```
# example of TFIDF dense matrix

Decided to start with simplest model for purposes of starting to engineer the features, tested the above vectorizers with a logistic regression:

```python

count_vect_score = cross_val_score(lr, X_train_counts, y_train, cv=5)

count_vect_score.mean()

# 0.9051418250935577

tf_vect_score = cross_val_score(lr, tfidf, y_train, cv=5)

tf_vect_score.mean()

# 0.8924947519654662

```

Then proceeded with `CountVectorizer()` and wanted to check on the range of n-grams to see how these scores might change with bigrams and above:

```python

def count_vec_ngram(params, X_train, y_train):
    cvec_p = CountVectorizer(ngram_range=(params))
    cvec_p.fit(X_train)
    X_train_cvec_p = cvec_p.transform(X_train)
    count_vect_score = cross_val_score(lr, X_train_cvec_p, y_train, cv=5)
    return count_vect_score.mean()

params = [(1,1), (1,2), (1,3), (1,4)] 

```

I was able to consistently get an accuracy score on the test set of between 90 and 91%.

I then wanted to try to compare this Logistic Regression model with a Naive Bayes model. Here are the top 50 classes for each target in the Naive Bayes model:

```python


Target: 0, name: @davideryu
Top 50 tokens:  ['http', 'la', 'org', 'city', 'ly', 'join', 'com', 'lacity', 'los', 'today', 'https', 'angeles', 'los angeles', 'bit', 'bit ly', 'lacd4', 'lacity org', 'community', 'thank', 'new', 'help', 'need', 'work', 'housing', 'http bit', 'learn', 'ow', 'ow ly', 'http ow', 'free', 'make', 'park', 'day', 'hollywood', 'great', 'council', 'info', 'support', 'program', 'davidryu', 'time', 'davidryu lacity', 'homelessness', 'rent', 'cd4', 'year', 'people', 'local', 'food', 'thanks']

Target: 1, name: @nithyavraman
Top 50 tokens:  ['la', 'city', 'com', 'people', 'replying', 'http', 'thread', 'housing', 'https', 'homeless', 'council', 'homelessness', 'need', 'just', 've', 'make', 'los', 'like', 'angeles', 'los angeles', 'help', 'nithyaforthecity', 'new', 'campaign', 'nithyaforthecity com', 'org', 'nithya', 'money', 'work', 'city council', 'right', 'latimes', 'time', 'rent', 'today', 'public', 'residents', 'latimes com', 'california', 'street', '000', 'want', 'services', 'nithya city', 'support', 'let', 'local', 'policy', 'workers', 'don']

```

Looking at these classes, I can see I need to adjust the stop words. I noticed there were many terms associated with links (http, com, bit, ly), as well as the endings of contractions (ve, re, ll) so I decided the English stop word list in Sklearn was not very effective for this use, so I ended up customizing the NLTK corpus stop word list to include these terms and had much more lexically interesting and sensible results.

Here below is a comparison between ROC curves for Naive Bayes and Logistic Regression.

```python

roc_auc_score for Naive Bayes:  0.940365682137834
roc_auc_score for Logistic Regression:  0.957597812390528

```

# Graph

I then continued to proceed with the Logistic Regression model but wanted to try optimizing with Stochastic Gradient Descent. The `SGDClassifier()` by default uses a linear Support Vector Classifier, but you can also use Logistic Regression. I also wanted to further tune the hyperparameters, so I employed the `GridSearchCV()` module:

```python

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='log')),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_features': (1000, 2000, 3000, 5000),
    'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),  # unigrams or bigrams
    'vect__stop_words': ('english', cust_stop),
    'clf__max_iter': (20,),
    'clf__alpha': (0.00001, 0.000001),
    'clf__penalty': ('l2', 'elasticnet'),

```

And got the following results:

```python

Best score: 0.953
Best parameters set:
	clf__alpha: 1e-05
	clf__max_iter: 20
	clf__penalty: 'elasticnet'
	vect__max_df: 1.0
	vect__max_features: 5000
	vect__ngram_range: (1, 1)

```

I was then able to get a test set score of `0.943579766536965`.

Naturally it is getting the best results with the most features, but I because this came back with n-gram range of (1,1) instead of the (1,2) I found earlier, I first wanted to see how accuracy changed not only with n-gram range, but also number of features:

# graph

I also wanted to use PCA to narrow down the feature set and identify where I can still get good accuracy with less features. I ran PCA on the entire set of Tweets, so ended up with 8223 features and plotted a scree plot for explained variance to see the inflection point:

#Graphs

Part of what interested me in doing this NLP project was to see if I could quantify or demonstrate any shift in the political discourse throughout the election. Particularly, David Ryu seemed to change his positions and language often throughout the election. The challenger Nithya was much further left on the political spectrum, and when he saw her success he emulated her language and positions. When that did not work to his favor, he then swung back to the right in an effort to swoon moderates.

I proceeded to use 3000 features for the rest of the tests as well as for looking at the principal components over time. I initially intended to look at Cosine Similarity to compare the tweets of the two candidates, but for the sake of visualization I went with PCA. I plotted the tweets along the first two principal components, as well as prominent terms with arrows corresponding to the loading within that principal component. I went back to August 2019, which is when Nithya Raman announced her candidacy:

PCA

Random Forest and out-of-the-box score

Same `CountVectorizer()` with 3000 features and bigrams. With 100 estimators, OOB score of 94%, and on the test set 92.6%.

Compared each model I'd explored, as well as some new ones:

```python

Validation result for Logistic Regression
LogisticRegression()

null accuracy: 65.42
accuracy score: 91.39
model is 25.97 more accurate than null accuracy

--------------------------------------------------------------------------------
Validation result for Linear SVC
LinearSVC()

null accuracy: 65.42
accuracy score: 94.70
model is 29.28 more accurate than null accuracy

--------------------------------------------------------------------------------
Validation result for Multinomial NB
MultinomialNB()
null accuracy: 65.42
accuracy score: 89.88
model is 24.46 more accurate than null accuracy

--------------------------------------------------------------------------------
Validation result for Random Forest
RandomForestClassifier()

null accuracy: 65.42
accuracy score: 93.24
model is 27.82 more accurate than null accuracy

--------------------------------------------------------------------------------
Validation result for AdaBoost
AdaBoostClassifier()

null accuracy: 65.42
accuracy score: 83.22
model is 17.80 more accurate than null accuracy

--------------------------------------------------------------------------------
Validation result for XGBoost
XGBClassifier(base_score=None, booster=None, colsample_bylevel=None,
              colsample_bynode=None, colsample_bytree=None, gamma=None,
              gpu_id=None, importance_type='gain', interaction_constraints=None,
              learning_rate=None, max_delta_step=None, max_depth=None,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              n_estimators=100, n_jobs=None, num_parallel_tree=None,
              random_state=None, reg_alpha=None, reg_lambda=None,
              scale_pos_weight=None, subsample=None, tree_method=None,
              validate_parameters=None, verbosity=None)

null accuracy: 65.42
accuracy score: 91.78
model is 26.36 more accurate than null accuracy

--------------------------------------------------------------------------------

```


XGBoost piqued my interest and I wanted to do more exploration on why it was so relatively low.

Moving ahead with XGBoost with max-depth of 2 and 30 estimators with 5-fold cross-validation (using same `CountVectorizer()`):

```python

Accuracy: 79.20% (1.05%)

                precision    recall  f1-score   support

           0       0.91      0.44      0.59       711
           1       0.77      0.98      0.86      1345

    accuracy                           0.79      2056
   macro avg       0.84      0.71      0.72      2056
weighted avg       0.81      0.79      0.77      2056


```
Not great! Let's do some hyperparameter tuning and see how we can improve that score:

```python

pipeline: ['vect', 'tfidf', 'clf']
parameters:
{'clf__max_depth': [2, 4, 8],
 'clf__n_estimators': [50, 100, 200, 400],
 'clf__subsample': [0.3, 0.5, 0.7],
 'vect__max_df': (0.5, 0.75),
 'vect__max_features': (1000, 2000, 3000),
 'vect__ngram_range': ((1, 1), (1, 2), (1, 3))}
Fitting 5 folds for each of 648 candidates, totalling 3240 fits

Best score: 0.931
Best parameters set:
	clf__max_depth: 8
	clf__n_estimators: 400
	clf__subsample: 0.7
	vect__max_df: 0.5
	vect__max_features: 2000
	vect__ngram_range: (1, 3)

```


Lastly I wanted to quickly try multiple classification with XGBoost so I scraped tweets from arguably the most famous tweeter: @realDonaldTrump himself.

Top terms: 'trump', 'team', 'great', 'vote'!

Used same vectorization pipeline and tested on an XGBoost model with max depth of 4 with 50 estimators:

```python

                   precision    recall  f1-score   support

      @davideryu       0.80      0.93      0.86      1345
   @nithyavraman       0.88      0.67      0.76       711
@realDonaldTrump       0.90      0.86      0.88       974

        accuracy                           0.84      3030
       macro avg       0.86      0.82      0.83      3030
    weighted avg       0.85      0.84      0.84      3030
    
```



# Miscellanea

I became interested in Zipf's law and wanted to see how this feature engineering changes the distribution of words compared to a predicted Zipf distribution.

I used the term frequencies from the TF-IDF vectorizer and plotted against a predicted Zipf line:

# Graph

* Fed Trump's tweets in vectorized format into the model trained on just Raman and Ryu and it overwhelmingly predicted Ryu. Difficult to say why or whether there is any credibility to the result as I didn't have much chance to explore further, but validated one of my motivations for this project to a certain degree:

```python

{0: 232, 1: 3663}

```
