# IMDB5000 
The dataset that I have worked with in this data analysis is available on [here](https://www.kaggle.com/deepmatrix/imdb-5000-movie-dataset/downloads/imdb-5000-movie-dataset.zip)


## First View 
At the beginning, we have an overview of how certain features may behave with one another. We plotted a simple Hexbin graph to investigate the relationship between the movie score and its duration as well as the gross.

### IMDB Score vs. Duration
[g1]: https://github.com/sztoo/kaggle/blob/master/imdb/screenshots/score_duration.png "Score vs. Duration"
![score_duration][g1]

### IMDB Score vs. Gross
[g2]: https://github.com/sztoo/kaggle/blob/master/imdb/screenshots/score_gross.png "Score vs. Gross"
![score_gross][g2]

## Moving On
We then investiage on how correlated are the features with one another. We adopted a common technique called Pearson's Correlation to see how correlations between the features.

### Pearson's Correlation
[g3]: https://github.com/sztoo/kaggle/blob/master/imdb/screenshots/correlation.png "Pearson's Correlation"
![p_cor][g3]

A few worth noticing trend here is the correlation between the *actor_1_facebook_likes* with the *cast_total_facebook_likes* as well as the *num_critic_for_reviews* and *movie_facebook_likes*. These features in the dimensions are positively correlated which indicates that actor1 may be a super star that brought the fame and attention to the movie itself, resulting in the increase of the Facebook likes on the movie page, or vise versa.

## PCA 
We then performed a PCA to see if there is any distincitve obvious clusters that we could group using KMeans (not implemented yet).

### Projection of First 2 PCA Components
[g4]: https://github.com/sztoo/kaggle/blob/master/imdb/screenshots/pca_largest_2.png "First 2 PCA Components"
![pca][g4]
We don't exactly see any obvious clusterings here, but with the explained variance, we can see that it is estimated around 9 components to capture 90% of the variances.
