#!/usr/bin/env python
# coding: utf-8

# # MOVIE RECOMMENDATION SYSTEM

# ### 1) DATA IMPORT

# In[87]:


import numpy as np                                                      # importing numpy and pandas python library
import pandas as pd


# In[88]:


movies = pd.read_csv("/Users/ezhilan/Desktop/Datasets/tmdb_5000_movies.csv")             # reading datasets using 
credits = pd.read_csv("/Users/ezhilan/Desktop/Datasets/tmdb_5000_credits.csv")           # pd.read_csv function 


# In[89]:


movies.head()                                                           # displays first few rows of the dataframe 


# In[90]:


credits.head()                                                          # displays first few rows of the dataframe 


# In[91]:


movies = movies.merge(credits,on='title')                      

# merging cast and crew column from credits dataset to movies dataset on the column title
# The merged dataset has combined both movie details and their respective cast & crew details 
# based on the movie title


# In[92]:


movies.head()


# In[93]:


movies.info()                                                        # fetches concise summary of the DataFrame 


# ## 2) DATA MANIPULATION

# In[94]:


# id
# title 
# overview 
# genres
# keywords
# cast
# crew

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

# reassigns the movies DataFrame to only include 
# specific columns: 'movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', and 'crew'.


# In[95]:


movies.head()


# In[96]:


movies.isnull().sum()           # used to count the number of missing values in each column of the movies DataFrame 


# In[97]:


movies.dropna(inplace=True)                               # used to remove any rows in the movies DataFrame 
                                                          # that contain missing values in any of its columns.


# In[98]:


movies.isnull().sum()           # used to count the number of missing values in each column of the movies DataFrame


# In[99]:


movies.duplicated().sum()                            # checks for and counts duplicate rows in the movies DataFrame


# In[100]:


movies.iloc[0].genres      # extracts the 'genres' value of the first row in the movies DataFrame using iloc method


# In[101]:


# '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'

# ['Action','Adventure','Fantasy','SciFi']


# In[102]:


def convert(obj):                                   
    L = []                                        # appends action,adventure,fantasy into new list
    for i in obj:
        L.append(i['name'])
    return L


# In[103]:


convert('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[104]:


import ast

ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[105]:


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):                          # # appends action,adventure,fantasy into new list
        L.append(i['name'])
    return L


# In[106]:


movies['genres'].apply(convert)


# In[107]:


movies['genres'] = movies['genres'].apply(convert)


# In[108]:


movies.head()


# In[109]:


movies['keywords'].apply(convert)


# In[110]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[111]:


movies.head()


# In[112]:


movies['cast'][0]


# In[113]:


def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L


# In[114]:


movies['cast'].apply(convert3)


# In[115]:


movies['cast'] = movies['cast'].apply(convert3)


# In[116]:


movies.head()


# In[117]:


movies['crew'][0]


# In[118]:


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i ['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[119]:


movies['crew'].apply(fetch_director)


# In[120]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[121]:


movies.head()


# In[122]:


movies['overview'][0]


# In[123]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())    # The lambda function takes a string x
                                                                     # and splits it into a list of words


# In[124]:


movies['overview'][0]


# In[125]:


movies.head()


# In[126]:


movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])         # removes spaces from each string within 
                                                                       # the lists of the 'genres' column.


# In[127]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])


# In[128]:


movies.head()


# In[129]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[130]:


movies.head()


# In[131]:


new_df = movies[['movie_id','title','tags']]


# In[132]:


new_df


# In[133]:


new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))       # it transforms a list of strings 
                                                                   # into a single string where the original - 
                                                                   # strings are separated by spaces


# In[134]:


new_df.head()


# In[135]:


import nltk                                                   # imports NaturalLanguageToolKit library in python


# In[136]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[137]:


def stem(text):
    y = []
    
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)


# In[138]:


new_df['tags'].apply(stem)


# In[139]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[140]:


new_df['tags'][0]


# In[141]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())            # it converts each string in the 'tags'column
                                                                     # to its lowercase version.


# In[142]:


new_df.head()


# In[143]:


new_df['tags'][0]


# In[144]:


new_df['tags'][1]


# In[145]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000,stop_words='english')


# In[146]:


cv = CountVectorizer()
vectors = cv.fit_transform(new_df['tags']).toarray()


# In[147]:


vectors


# In[148]:


vectors [0]


# In[149]:


cv.get_feature_names_out()


# In[150]:


['loved','loving','love']
['love','love','love']


# In[151]:


ps.stem('loved')


# In[152]:


ps.stem('loving')


# In[153]:


stem('in the 22nd century, a paraplegic marine is dispatched to the moon pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. action adventure fantasy sciencefiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d samworthington zoesaldana sigourneyweaver jamescameron')


# In[154]:


from sklearn.metrics.pairwise import cosine_similarity  # used for comparing the similarity of vectors
                                                        # representing documents, sentences, or other textual data


# In[155]:


cosine_similarity(vectors).shape


# In[156]:


similarity = cosine_similarity(vectors)


# In[157]:


similarity


# In[158]:


similarity[1]


# In[159]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[174]:


def recommend(movie):
    movie_index = new_df[new_df['title']== movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[175]:


new_df[new_df['title']=='Avatar']


# In[176]:


recommend('Avatar')


# In[177]:


recommend('Batman Begins')


# In[ ]:





# In[164]:


import pickle


# In[165]:


pickle.dump(new_df,open('movies.pkl','wb'))


# In[166]:


new_df['title'].values


# In[167]:


new_df.to_dict()


# In[168]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[169]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




