import pandas as pd
import numpy as np
# Get the Data
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=column_names)
print(df.head())

movie_titles = pd.read_csv('Movie_Id_Titles')
print(movie_titles.head())
# Merge
df = pd.merge(df, movie_titles, on='item_id')
print(df.head())

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

print(df.groupby('title')['rating'].mean().sort_values(ascending=False).head())
print(df.groupby('title')['rating'].count().sort_values(ascending=False).head())
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
print(ratings.head())

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings['num of ratings'] = df.groupby('title')['rating'].count()
print(ratings.head())

import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))
ratings['num of ratings'].hist(bins=70)
plt.xlabel('Number of Ratings')
plt.ylabel('Frequency')
plt.title('Histogram of Number of Ratings per Title')
plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))
ratings['rating'].hist(bins=70)
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.title('Histogram of Average Ratings per Title')
plt.show()


sns.jointplot(x='rating', y='num of ratings', data=ratings, alpha=0.5)
plt.show()

moviemat = df.pivot_table(index='user_id', columns='title', values='rating')
print(moviemat.head())

print(ratings.sort_values('num of ratings', ascending=False).head(10))

print(ratings.head())
starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']
print(starwars_user_ratings.head())

similar_to_starwars = moviemat.corrwith(starwars_user_ratings, drop=True)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings, drop=True)
corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)
print(corr_starwars.head())

print(corr_starwars.sort_values('Correlation', ascending=False).head(10))

# First, join 'num of ratings' from ratings DataFrame
corr_starwars = corr_starwars.join(ratings['num of ratings'])

# Now filter by 'num of ratings' > 100 and sort
filtered = corr_starwars[corr_starwars['num of ratings'] > 100]
top_similar = filtered.sort_values('Correlation', ascending=False).head()

print(top_similar)

corr_liarliar = pd.DataFrame(similar_to_liarliar, columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
top_liarliar = corr_liarliar[corr_liarliar['num of ratings'] > 100] \
    .sort_values('Correlation', ascending=False).head()

print(top_liarliar)


