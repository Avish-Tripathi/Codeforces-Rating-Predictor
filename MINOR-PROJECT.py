#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')


# ## Dataset
# 

# In[2]:


# READING THE DATASET
df=pd.read_csv("file2.csv")
df.head(10)


# ## Describe Data

# In[3]:


# DATATYPES IN THE DATASET
df.dtypes


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


# DELETING THE 'HANDLE' COLUMN FROM THE DATASET
df=df.drop("handle",axis=1)
df.head()


# # Visualizing the data

# In[7]:


df.plot(x='problem_count',y="max_rating",kind="scatter");


# In[8]:


df.plot(x='friends_count',y="max_rating",kind="scatter");


# In[9]:


df['friends_count'].plot.hist(bins = [0,50,100,200,300,400,500,600,700]);


# In[10]:


df['contest_count'].plot.hist();


# In[11]:


df.plot(x='problem_count',y="rating",kind="scatter");


# In[12]:


df.plot(x='problem_count',y="friends_count",kind="scatter");


# In[14]:


plt.figure(figsize=(10, 6))
sns.boxplot(x=df["rating"])


# In[15]:


plt.figure(figsize=(10, 6))
sns.boxplot(x=df["max_rating"])


# ## Removing The Outliers

# In[16]:


# Outliers in Rating
df['rating'].plot.hist(bins=50)
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Rating Graph')


# In[17]:


# Finding the Min and Max Value
mi=df['rating'].min()
mx=df['rating'].max()
print('min:',mi)
print('max:',mx)


# In[18]:


df['rating'].describe()


# In[19]:


# Plotting the Bell Curve
from scipy.stats import norm
df['rating'].plot.hist(bins=50,density=True)
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Rating Graph')
rng=np.arange(df['rating'].min(),df['rating'].max())
plt.plot(rng,norm.pdf(rng,df['rating'].mean(),df['rating'].std()))


# In[20]:


df['zscore']=((df.rating-df.rating.mean()))/df.rating.std()
df.head()


# In[21]:


df[df['zscore']>3]


# In[22]:


df=df[(df['zscore']<3) & (df['zscore']>-3)]
df=df.drop("zscore",axis=1)
df


# In[23]:


# Outliers in Friends_Count
df['friends_count'].plot.hist(bins=20)
plt.xlabel('friends_count')
plt.ylabel('Count')
plt.title('friends_count Graph')


# In[24]:


df['friends_count'].describe()


# In[25]:


# Plotting the Bell Curve
from scipy.stats import norm
df['friends_count'].plot.hist(bins=20,density=True)
plt.xlabel('friends_count')
plt.ylabel('Count')
plt.title('friends_count Graph')
rng=np.arange(df['friends_count'].min(),df['friends_count'].max())
plt.plot(rng,norm.pdf(rng,df['friends_count'].mean(),df['friends_count'].std()))


# In[26]:


df['zscore']=((df.friends_count-df.friends_count.mean()))/df.friends_count.std()
df=df[(df['zscore']<3) & (df['zscore']>-3)]
df=df.drop("zscore",axis=1)
df


# In[27]:


# Outliers in problem_count
df['problem_count'].plot.hist(bins=20)
plt.xlabel('problem_count')
plt.ylabel('Count')
plt.title('problem_count Graph')


# In[28]:


# Plotting the Bell Curve
from scipy.stats import norm
df['problem_count'].plot.hist(bins=20,density=True)
plt.xlabel('problem_count')
plt.ylabel('Count')
plt.title('problem_count Graph')
rng=np.arange(df['problem_count'].min(),df['problem_count'].max())
plt.plot(rng,norm.pdf(rng,df['problem_count'].mean(),df['problem_count'].std()))


# In[29]:


df['zscore']=((df.problem_count-df.problem_count.mean()))/df.problem_count.std()
df=df[(df['zscore']<3) & (df['zscore']>-3)]
df=df.drop("zscore",axis=1)
df


# In[30]:


# Outliers in max_rating
df['max_rating'].plot.hist(bins=20)
plt.xlabel('max_rating')
plt.ylabel('Count')
plt.title('max_rating Graph')


# In[31]:


# Plotting the Bell Curve
from scipy.stats import norm
df['max_rating'].plot.hist(bins=20,density=True)
plt.xlabel('max_rating')
plt.ylabel('Count')
plt.title('max_rating Graph')
rng=np.arange(df['max_rating'].min(),df['max_rating'].max())
plt.plot(rng,norm.pdf(rng,df['max_rating'].mean(),df['max_rating'].std()))


# In[32]:


df['zscore']=((df.max_rating-df.max_rating.mean()))/df.max_rating.std()
df=df[(df['zscore']<3) & (df['zscore']>-3)]
df=df.drop("zscore",axis=1)
df


# # MODEL

# ## Random Forest

# In[33]:


X=df.iloc[:,:-1].values
Y=df.iloc[:,4]
Y


# In[34]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test,=train_test_split(X,Y,test_size=1/3,random_state=0)


# In[35]:


from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(x_train,y_train)
ans=regressor.predict(x_test)


# In[36]:


ans


# In[37]:


from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, ans))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, ans))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, ans)))


# In[38]:
pickle.dump(regressor,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
# In[ ]:




