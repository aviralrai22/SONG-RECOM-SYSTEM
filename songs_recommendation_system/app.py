#here we will use streamlit library for fronted design
from model import recommend
from sklearn.neighbors import NearestNeighbors
import streamlit as st
import pandas as pds

st.title('Songs recommendation system')

    
# need to fetch songs pickle file
import pickle as pi
songs_list=pi.load(open('songs.pkl','rb'))
#make a datarame of the file 
songs=pds.DataFrame(songs_list)

#load the list of the album name into the selectform
selected_song=st.selectbox("select the song",songs['album_name'].values)
if st.button('Recommend'):
    recom=recommend(selected_song)
    st.write(recom)
