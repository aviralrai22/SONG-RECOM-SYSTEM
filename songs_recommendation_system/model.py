import numpy as np 
import pandas as pd
songs=pd.read_csv("E:\machine_learning _projects\songs_recommendation_system\dataset.csv")


#to see all the columns without truncation
# pd.set_option("display.max_columns", None)
# print(songs.head())

#removing the null values
# print(songs.isnull().sum()) #check for the null columns

#removing the column name with the null values with mode
for col in ['artists', 'album_name']:
    songs[col] = songs[col].fillna(songs[col].mode()[0])

# print(songs.isnull().sum()) #check for the null columns

# print(songs.columns)
#drop unnecessary columns
songs.drop(["track_id", "track_name","duration_ms","explicit","key","mode","time_signature","track_genre"], axis=1, inplace=True)

#remove the duplicate songs with same name may be it has different versions
songs=songs.drop_duplicates(subset=['album_name'],keep="first",ignore_index=True)
# print(songs.columns)



#normalisation of the dataframe
from sklearn.preprocessing import MinMaxScaler




# Select numeric columns
cols_to_normalize = ["popularity", "loudness","instrumentalness","tempo"]

scaler = MinMaxScaler()
songs[cols_to_normalize] = scaler.fit_transform(songs[cols_to_normalize])













# Step 1: Count frequency (relative frequency, 0-1)
artist_freq = songs['artists'].value_counts() / len(songs)

# Step 2: Map the frequencies to a new column
songs['artists_freq'] = songs['artists'].map(artist_freq)
songs.drop("artists",inplace=True,axis=1)


print(songs["album_name"])
#normalisation of the column artists frequency as it is too low


# Select numeric columns
cols_to_normalize = ["artists_freq"]

scaler = MinMaxScaler()
songs[cols_to_normalize] = scaler.fit_transform(songs[cols_to_normalize])
print(songs)

#training the model 
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Example: preprocessed numeric features
# Replace with your dataset
X = songs[["artists_freq","tempo","popularity","energy","loudness","speechiness","acousticness","instrumentalness","liveness","valence"]].values



# Fit KNN
#making knn ready for the use of query of the song total 5 nearest neighbours will be given
knn = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')  # top 5 + self
knn.fit(X)

# Find top N neighbors for a given song
#song_name = 'The Lion King' # index of the song
def recommend(song_name):
  song_idx=songs[songs['album_name']==song_name].index[0]

#caliberate the distances and indices of nearest neighbours return 2 tuples
  distances, indices = knn.kneighbors([X[song_idx]])

# Get song names (excluding itself)
  recommended = songs.iloc[indices[0][1:]]["album_name"]
  return recommended

#need to save the list of the songs and send to web.py
import pickle as pi
#need to send songs data frame as an dictionary as dataframe cant be send
pi.dump(songs.to_dict(),open('songs.pkl','wb'))



