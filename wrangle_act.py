#!/usr/bin/env python
# coding: utf-8

# # Importing all the libraries

# In[1]:


import pandas as pd
import numpy as np
import requests
import json
import os
from IPython.display import Image
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from datetime import datetime
warnings.filterwarnings('ignore')


# # Gather Data

# ### Import WeRateDogs Twitter Archive

# In[2]:


#show the data in dataset csv file twitter archives

df_twitter_archives = pd.read_csv('twitter-archive-enhanced-2.csv')


# ### Import Tweet Image Predictions

# In[3]:


# Download the image prediction file using the link provided to Udacity students

url = 'https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv'
images = requests.get(url, allow_redirects=True)

open('image_predictions.tsv', 'wb').write(images.content)


# In[4]:


#check the content of the image predection

images.content


# In[5]:


#Show the data in dataset file

df_image_predictions = pd.read_csv("image-predictions.tsv", sep = '\t')


# ### import Twitter API & JSON file

# In[6]:


#show the data in dataset txt file tweeter-json.txt

df_tweet_json = pd.read_json('tweeter-json.txt',lines = True)


# # Assessing the Data

# ### Display all Three datasets seperetly

# In[7]:


df_twitter_archives


# In[8]:


df_image_predictions


# In[9]:


df_tweet_json


# ## Programmatic assessment

# ### Using Pandas functions methods to assess the data.

# In[10]:


df_twitter_archives.head()


# In[11]:


df_twitter_archives.info()


# In[12]:


df_twitter_archives.shape


# In[13]:


df_twitter_archives.describe()


# In[14]:


df_image_predictions.head()


# In[15]:


df_image_predictions.shape


# In[16]:


df_image_predictions.info()


# In[17]:


df_tweet_json.head()


# In[18]:


df_tweet_json.shape


# In[19]:


df_tweet_json.info()


# In[20]:


df_tweet_json.describe()


# ### Archives Dataframe Assessing

# In[21]:


df_twitter_archives.columns


# In[22]:


df_twitter_archives.name.value_counts()


# In[23]:


sum(df_twitter_archives['tweet_id'].duplicated())


# In[24]:


df_twitter_archives['rating_numerator'].value_counts()


# In[25]:


df_twitter_archives['rating_denominator'].value_counts()


# In[26]:


df_twitter_archives['source'].value_counts()


# ### Image Dataframe Assessing

# In[27]:


df_image_predictions.columns


# In[28]:


df_image_predictions['p1'].value_counts()


# In[29]:


df_image_predictions['p2'].value_counts()


# In[30]:


df_image_predictions['p3'].value_counts()


# In[31]:


Image(url = 'https://pbs.twimg.com/media/CT42GRgUYAA5iDo.jpg')


# In[32]:


Image(url = 'https://pbs.twimg.com/media/DGBdLU1WsAANxJ9.jpg')


# ### Tweet Dataframe Assessing

# In[33]:


df_tweet_json.info()
df_tweet_json.describe()


# # Clean

# ## Quality

# ### Define

# - Making a copy from all three dataframes to clean them
# - Check notnull in retweeted_status_id
# - Select the needed columns from tweet count (tweet.json) dataframe
# - Merge the clean versions of df_twitter_archive, df_image_predictions, and df_tweet_json dataframes
# - Check the new Dataframe
# - Convert tweet_id from an integer to a string
# - Capitalize P1, P2, P3
# - Fillt doggo, floofer, pupper, puppo Columns
# - Create one column for the various dog types: Doggo, Floofer, Pupper, Puppo
# - Convert rating_numerator from an integer to a Float
# - Convert rating_denominator from an integer to a Float
# - Convert the timestamp to correct datetime format
# - Create one column for the various dog types: P1,P2,P3
# - Drop rows that has prediction_list equal error
# - Remove columns no longer needed

# ## Tidiness

# - Change tweet_id to type int to merge with the other Dataframes
# - All Dataframes should be one dataset

# ### Quality Issue 1: Making a copy from all three dataframes to clean them

# #### Define

# Making a copy from all three dataframes to clean them

# #### Code

# In[34]:


# Make copies of dataframes to clean 
dfclean_twitter_archives = df_twitter_archives.copy()
dfclean_image_predictions = df_image_predictions.copy()
dfclean_tweet_json = df_tweet_json.copy()


# #### Test

# ### Check the copied dataframe

# In[35]:


dfclean_twitter_archives.sample(10)
dfclean_image_predictions.sample(10)
dfclean_tweet_json.sample(10)


# In[36]:


dfclean_image_predictions.sample(10)


# In[37]:


dfclean_tweet_json.sample(10)


# ### Quality Issue 2: Delete retweets

# #### Define

# Delete retweets

# #### Code

# In[38]:


#Check notnull in retweeted_status_id

#dfclean_twitter_archives.drop(dfclean_twitter_archives[dfclean_twitter_archives['retweeted_status_id'].notnull()])
dfclean_twitter_archives = dfclean_twitter_archives[dfclean_twitter_archives['retweeted_status_id'].isnull()]
sum(dfclean_twitter_archives["retweeted_status_id"].notnull())


# #### Test

# In[39]:


dfclean_twitter_archives.head()


# ### Quality Issue 3: Select the needed columns from tweet count Dataframe

# #### Define

# Select the needed columns from tweet count Dataframe (id, retweet_count, favorite_count)

# #### Code

# In[40]:


dfclean_tweet_json = dfclean_tweet_json[["id", "retweet_count", "favorite_count"]]


# #### Test

# In[41]:


dfclean_tweet_json.sample(10)


# In[42]:


dfclean_tweet_json.info()


# ### Quality Issue 4: Merge all three Dataframes in one Dataframe

# #### Define

# Merge all three Dataframes in one Dataframe called df

# #### Code

# In[43]:


#merge all three dataframes in one dataframe called df

df = pd.concat([dfclean_twitter_archives, dfclean_image_predictions, dfclean_tweet_json], join ='outer',axis=1)
df1= df


# #### Test

# #### Checking the new Dataframe

# In[44]:


df.head()


# In[45]:


df.shape


# In[46]:


df.columns


# In[47]:


df.info()


# In[48]:


df.describe()


# In[49]:


df.sample(10)


# ### Quality Issue 5: Convert tweet_id Column to Str

# #### Define

# Convert tweet_id Column to str insted of int
# 

# #### Code

# In[50]:


#Convert tweet_id Column to str insted of int

dfclean_twitter_archives['tweet_id'] = dfclean_twitter_archives['tweet_id'].astype(str)
dfclean_image_predictions['tweet_id'] = dfclean_image_predictions['tweet_id'].astype(str)


# #### Test

# In[51]:


dfclean_twitter_archives.info()


# In[52]:


dfclean_image_predictions.sample(10)


# ### Quality Issue 6: Capitalize P1, P2, P3

# #### Define

# Capitalize each name of the Dog names in p1, p2, p3 Columns

# #### Code

# In[53]:


#Capitalize p1,p2,p3

dfclean_image_predictions['p1'] = dfclean_image_predictions['p1'].str.title()
dfclean_image_predictions['p2'] = dfclean_image_predictions['p2'].str.title()
dfclean_image_predictions['p3'] = dfclean_image_predictions['p3'].str.title()


# #### Test

# In[54]:


dfclean_image_predictions.head()


# In[55]:


dfclean_image_predictions.info()


# ### Quality Issue 7: Filter doggo, floofer, pupper, puppo Columns

# #### Define

# Filter doggo, floofer, pupper, puppo Columns to get the values of Doggo type 

# #### Code

# In[56]:


#fillt doggo column

df['doggo'].sample(50)
doggofilt = df['doggo'] == 'doggo'


# #### Test

# In[57]:


df[doggofilt]


# #### Define

# Filter doggo, floofer, pupper, puppo Columns to get the values of Floofer type 

# #### Code

# In[58]:


#fillt floofer column

df['floofer'].sample(50)
flooferfilt = df['floofer'] == 'floofer'


# #### Test

# In[59]:


df[flooferfilt]


# #### Define

# Filter doggo, floofer, pupper, puppo Columns to get the values of Pupper type

# #### Code

# In[60]:


#fillt pupper column

df['pupper'].sample(50)
pupperfilt = df['pupper'] == 'pupper'


# #### Test

# In[61]:


df[pupperfilt]


# #### Define

# Filter doggo, floofer, pupper, puppo Columns to get the values of Puppo type

# #### Code

# In[62]:


#fillt puppo column

df['puppo'].sample(50)
puppofilt = df['puppo'] == 'puppo'


# #### Test

# In[63]:


df[puppofilt]


# In[64]:


#get the value of dogs type (doggo,floofer,pupper,puppo)

df[['doggo','floofer','pupper','puppo']].value_counts()


# In[65]:


#get the value of dogs type (pupper) in df

df[['pupper']].value_counts()


# In[66]:


#get the value of dogs type (puppo) in df

df[['puppo']].value_counts()


# In[67]:


#get the value of dogs type (doggo) in df

df[['doggo']].value_counts()


# In[68]:


#get the value of dogs type (floofer) in df

df[['floofer']].value_counts()


# ### Quality Issue 8: Create one column for the various dog types: Doggo, Floofer, Pupper, Puppo

# #### Define

# Create one column for the various dog types: Doggo, Floofer, Pupper, Puppo and Drop The Doggo, Floofer, Pupper, Puppo Columns From The Dataframe 

# #### Code

# In[69]:


df.columns


# In[70]:


# Create one column for the various dog types: Doggo, Floofer, Pupper, Puppo

df['dog_stage'] = df['text'].str.extract('(doggo|floofer|pupper|puppo)')
df = df.drop(['doggo','floofer','pupper','puppo'],axis = 1)


# #### Test

# In[71]:


df['dog_stage'].value_counts()


# In[72]:


df.info()


# In[73]:


df.columns


# ### (Optional)filter new dataframe to get dog types columns without na and call it df_new

# In[74]:


#filter new dataframe to get dog types columns without na and call it df_new

filt = (df1['puppo'] == 'puppo') | (df1['pupper'] == 'pupper') | (df1['floofer'] == 'floofer') | (df1['doggo'] == 'doggo')
df_new = df1[filt]
#df[filt]["puppo","pupper"]
df_new.columns
df_new[['doggo','floofer','pupper','puppo']].value_counts()
#dfn[["tweet_id","in_reply_to_status_id"]]


# In[75]:


#get the value of dogs type (pupper) in df_new

df_new[['pupper']].value_counts()


# In[76]:


#get the value of dogs type (puppo) in df_new

df_new[['puppo']].value_counts()


# In[77]:


#get the value of dogs type (doggo) in df_new

df_new[['doggo']].value_counts()


# In[78]:


#get the value of dogs type (floofer) in df_new

df_new[['floofer']].value_counts()


# In[79]:


df['text']


# In[80]:


df_new


# In[81]:


df_new.info()


# In[82]:


df_new.head()


# In[83]:


df_new.columns


# In[84]:


df_new.shape


# In[ ]:





# In[ ]:





# ### Quality Issue 9: Convert rating_numerator from an integer to a Float

# #### Define

# Convert rating_numerator from an integer to a Float

# #### Code

# In[85]:


df["rating_numerator"] = df["rating_numerator"].astype(float)


# #### Test

# In[86]:


df.info()


# ### Quality Issue 10: Convert rating_denominator from an integer to a Float

# #### Define

# Convert rating_denominator from an integer to a Float

# #### Code

# In[87]:


df["rating_denominator"] = df["rating_denominator"].astype(float)


# #### Test

# In[88]:


df.info()


# ### Quality Issue 11: Convert the timestamp to correct datetime format

# #### Define

# Convert the timestamp to correct datetime format

# #### Code

# In[89]:


df["timestamp"] = pd.to_datetime(df["timestamp"])


# #### Test

# In[90]:


df.info()


# ### Quality Issue 12: Create one column for the various dog types: P1,P2,P3

# In[91]:


df["dog_types"] = "null"


# #### Define

# Create one column for the various dog types: P1,P2,P3

# #### Code

# In[92]:


df


# In[93]:


df[['p1']].value_counts()


# In[94]:


df[['p2']].value_counts()


# In[95]:


df[['p3']].value_counts()


# In[96]:


#creat a new dog types column using p1,p2,p3

dog_type = []
def dog_types(df):
    if df['p1_dog'] == True :
        dog_type.append(df["p1"])
    elif df['p2_dog'] == True :
        dog_type.append(df["p2"])
    elif df['p3_dog'] == True :
        dog_type.append(df["p3"])  
    else :
        dog_type.append("Error")  
df.apply(dog_types, axis = 1)        
display(dog_type)    
df['dog_types'] = dog_type


# #### Test

# In[97]:


df.sample(10)


# In[98]:


df.info()


# ### Quality Issue 13: Drop rows that has prediction_list equal error

# #### Define

# Drop rows that has prediction_list equal error after creating dog_types Column

# #### Code

# In[99]:


#drop rows that has prediction_list 'error'

df.drop(df.loc[df["dog_types"] == "Error"].index ,inplace = True)
df[df["dog_types"] == "Error"]


# #### Test

# In[100]:


df.info()


# In[101]:


df.sample(10)


# In[102]:


df.columns


# ### Quality Issue 14: Drop unuseful Columns in df

# #### Define

# Drop unuseful Columns From the Dataframe That will not help anymore  

# #### Code

# In[103]:


#drop unuseful columns in df

df = df.drop(['source','in_reply_to_status_id','in_reply_to_user_id','retweeted_status_id','retweeted_status_user_id', 'retweeted_status_timestamp', 'expanded_urls'],axis = 1)


# #### Test

# In[104]:


df.columns


# # Storing Cleaned Data in a new Dataset call

# In[105]:


# Storing the new twitter_dogs df to a new csv file

df.to_csv('twitter_archive_master.csv', encoding='utf-8', index=False)


# # Analyzing, and Visualizing Data
# 

# - Visualize for the Most Rated Dog Type
# - Visualize for the Most common Dog name
# - Visualize for the Most common Dog name without nan
# - Favorite Count Distribution Plot
# - Retweet Count Distribution
# - The Relationship Between Favorites and Retweets Count
# - A Heatmap Visualize

# In[106]:


df['dog_types'].value_counts()


# ### Visualize for the Most Rated Dog Type

# In[107]:


#Visualize for the Most Rated Dog Type

dog_types = df.groupby('dog_types').filter(lambda x: len(x) >= 25)
dog_types['dog_types'].value_counts().plot(kind = 'bar')
plt.title('Bar Chart of the Most Rated Dog Type')
plt.xlabel('Count')
plt.ylabel('Type of dog')


# In[108]:


df["name"].value_counts()


# ### Visualize for the Most common Dog name

# In[109]:


#Visualize for the Most common Dog name

df["name"].value_counts().iloc[0:9].plot(kind= 'bar')
plt.title('Bar Chart for the Most common Dog name')
plt.xlabel('Count')
plt.ylabel('name of dog')


# In[110]:


df_name = df.drop(df.loc[df["name"] == "None"].index)
df_name['name'].value_counts()
df_name['name'].value_counts().iloc[0:8]


# ### Visualize for the Most common Dog name without nan

# In[111]:


#Visualize for the Most common Dog name without nan

df_name = df.drop(df.loc[df["name"] == "None"].index)
df_name['name'].value_counts().iloc[0:8].plot(kind= 'bar')
plt.title('Bar Chart for the Most common Dog name')
plt.xlabel('Count')
plt.ylabel('name of dog')


# In[112]:


df.groupby('dog_types')['favorite_count'].describe()


# In[113]:


df.head()


# ### Favorite Count Distribution Plot

# In[114]:


#Favorite Count Distribution Plot

plt.figure(figsize =(10,10))
sns.distplot(df["favorite_count"],  label = 'Favorites Count')
plt.title("Favorites Count")
plt.xlabel('Favorites Count')
plt.show()


# ### Retweet Count Distribution

# In[115]:


#Retweet Count Distribution

plt.figure(figsize =(10,10))
sns.distplot(df['retweet_count'], color = 'blue', label = 'Retweets Count')
plt.title("Retweets Count")
plt.xlabel('Retweets Count')
plt.show()


# In[116]:


df['retweet_count'].describe()


# In[117]:


df["favorite_count"].describe()


# ### The Relationship Between Favorites and Retweets Count

# In[118]:


#The Relationship Between Favorites and Retweets Count

plt.figure(figsize =(10,10))
g = sns.regplot(x=df['retweet_count'], y=df["favorite_count"])
plt.title("Favorites and Retweets Count")
plt.xlabel('Retweets Count')
plt.ylabel('Favorites Count')
plt.show()


# In[119]:


correlation = df.corr()


# In[120]:


correlation


# ### A Heatmap Visualize

# In[121]:


#A Heatmap Visualize

plt.figure(figsize =(10,10))
sns.heatmap(correlation,annot=True)
plt.title("The Correlation Between all Features")
plt.show()


# In[ ]:




