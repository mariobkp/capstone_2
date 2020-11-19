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

I propose using a combination of Twitter API, tweepy, getoldtweets3, python-twitter libraries to look at tweets possibly going back as far as August 7, 2019 when Nithya announced her campaign.

I would like to focus on NLP, exploring the tweets using unsupervised learning (K-means), perform sentiment analysis, as well as hopefully use the geographic data. Additionally would like to try out document clustering with Non-negative Matrix Factorization.

Could platform priorities be detected or predicted by the tweets of the candidates? Do Raman or Ryu supporters have common traits? Can sentiment be detected beyond positive/negative (i.e. pro-Raman/anti-Ryu or vice-versa)?
