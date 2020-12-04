# capstone_2

I propose an examination of Tweets and Twitter data regarding the LA city council election for council district (CD) 4 between incumbent David Ryu and Nithya Raman.

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



## Scraping Data
I was hoping to get clean Twitter data and metadata about the tweets in JSON format directly from the Twitter API in order to begin my project. Unfortunately it seems that fairly recently Twitter locked down much of their API and thus access to the data I needed. I did sign up for a Twitter developer account (and by process had to describe my project to Twitter to get approved), however with a free account I could only access Tweets from the past rolling 7 day period. Not a big enough window to even go back to the date of the election, let alone analyze the past Twitter activity of the candidates.

In locking down the API, Twitter also apparently cleaned up some "loopholes" that were utilized by some of the most common Python twitter scraping libraries, rendering them mostly useless in their current state.

Instead, I ended up using Selenium (highly recommend for ease of use, effectiveness, and it's is really fun) to create a program that can automatically scroll and scrape the most pertinent data (Twitter handle, timestamp, text, # of likes, # of comments, and # of retweets). I also parsed out and saved any emojis that I hoped to use (and maybe will in the future). I then saved the data in .csv format.

# Screenshot of selenium bot?

## Cleaning Data
Because of the way I wrote the program (basically a list of lists), it was easy to use Pandas and read the .csv files into DataFrames.


# Show dataframe sample here


# EDA

To start I just wanted to make some visuals of the data to see what I had.

# Graphs of numbers? tweets per month? average length of tweets (word count)? something with engagement?

Here are the most common words per candidate:

# Word cloud

And here are the most common hashtags:

# Word cloud

Pipeline #

Checked on effectiveness of `CountVectorizer()` and `TfidfVectorizer` on getting all this text into a usable numerical form.

# Graph

## Classification

First explored using a logistic regression model to classify the Tweets

Quick check on n-gram range, number of features

## Feature Engineering

As you've seen, calling cf.fit is not so challenging. The best solutions often leverage intelligent feature engineering. Noting the process of discovery through EDA to find relevant feature manipulations is a very powerful demonstration of you ability to understand problems in an analytic way. This is a great opportunity to expand and refine you data visualization prowess.

# Custom stop words

I noticed there were many terms associated with links, as well as the endings of contractions (ve, re, ll) so I decided the English stop word list in Sklearn was not very effective for this use, so I ended up customizing the NLTK corpus stop word list to include these terms and had much more lexically interesting and sensible results.

## Predictive Modeling

Perhaps the easiest to understand concept for the general population to understand in ML is the idea of a predictive analytic. Building a very effective model will be a great talking point for you when you are sharing your work during networking and job searching. You should put a good amount of thought into which measures of effectiveness you chose, the degree to which model interpretability is involve. In addition, you should have expectations which models you expect to perform well at this task, and determine if your intuition is upheld or not.

# Predict Nithya or Ryu

## Inferential Modeling

Interpreting models can be a great way of demonstrating your understanding of the mechanics of an ML model. Spending effort in understanding and communicating the how and why of your model behavior makes it much easier for stakeholders to accept model results. As such, this is not only an important aspect of capstones, but also a useful tool for thriving in your new role.




# Ryu change over time

Part of what interested me in doing this NLP project was to see if I could quantify or demonstrate any shift in the political discourse throughout the election. Particularly, David Ryu seemed to change his positions and language often throughout the election. The challenger Nithya was much further left on the political spectrum, and when he saw her success he emulated her language and positions. When that did not work to his favor, he then swung back to the right in an effort to swoon moderates.


# NMF
