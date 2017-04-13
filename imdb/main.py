# This data analysis uses the IMDB5000 datasets gotten from Kaggle
# https://www.kaggle.com/deepmatrix/imdb-5000-movie-dataset/downloads/imdb-5000-movie-dataset.zip

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

file = 'movie.csv'
data = pd.read_csv(file)
df = pd.DataFrame(data)

def sanatize(df):
  headers = []
  for col_name, val in df.iteritems():
    if not type(val[0]) == str: 
      headers.append(col_name)
  num_df = df[headers]

  # impute null values 
  num_df.fillna(value=0, axis=1, inplace=True)
  return(num_df)

def process(df):
  # standardization
  X = df.values
  X_std = StandardScaler().fit_transform(X)

  # overview of fitted data using hexbin
  comparison_list = [['imdb_score', 'duration'], ['imdb_score', 'gross']]
  for idx, l in enumerate(comparison_list):
    gridsize = 35 if idx == 1 else 45
    df.plot(y=l[0], x=l[1], kind='hexbin', gridsize=gridsize, colormap='cubehelix', title='{} vs {}'.format(l[0], l[1]))

  # Pearson's Correlation
  f, ax = plt.subplots(figsize=(10,10))
  plt.title("Pearson's Correlation of Dimensionality")
  hm = sns.heatmap(df.astype(float).corr(), linewidths=0.25, vmax=1, square=True, cmap="YlGnBu", linecolor="black")
  plt.yticks(rotation=0)
  plt.xticks(rotation=90)

  # Explained Variance
  mean_vec = np.mean(X_std, axis=0)
  cov_mat = np.cov(X_std.T)
  eig_val, eig_vec = np.linalg.eig(cov_mat)
  
  # (eig_val, eig_vec) tuples
  eig_pair = [(np.abs(eig_val[i], eig_vec[:,i]) for i in range(len(eig_val)))]
  eig_pair.sort(key = (lambda x: x[0]), reverse=True)
  total = sum(eig_val)
  exp_var = [(i/total)*100 for i in sorted(eig_val, reverse=True)] # individual explained variance 
  # cumulative explained variance 
  cum_exp_var = np.cumsum(exp_var)
  plt.figure(figsize=(8, 5))
  plt.bar(range(16), var_exp, alpha=0.3333, align='center', label='individual explained variance', color = 'g')
  plt.step(range(16), cum_var_exp, where='mid',label='cumulative explained variance')
  plt.ylabel('Explained variance ratio')
  plt.xlabel('Principal components')
  plt.legend(loc='best')

  # PCA
  # 90% of variance captured
  pca = PCA(n_components=9)
  x_9d = pca.fit_transform(X_std)
  plt.figure(figsize=(7,7))
  plt.scatter(x_9d[:,0], x_9d[:,1], c='goldenrod', alpha=0.5)
  plt.ylim(-10,30)
  plt.show()

def main(): 
  c_df = sanatize(df)
  process(c_df)
  
if __name__ == "__main__":
  main()