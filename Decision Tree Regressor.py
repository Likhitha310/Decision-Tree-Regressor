#!/usr/bin/env python
# coding: utf-8

# # Import the libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.tree import DecisionTreeRegressor


# # Import the dataset

# In[2]:


df = pd.read_csv('IceCreamData.csv')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


plt.scatter(df.Temperature, df.Revenue)
plt.xlabel('Temperature')
plt.ylabel('Revenue')
plt.title('Temperature V/s Revenue')


# In[9]:


sns.heatmap(df.corr(), annot=True, cmap='Greens')


# In[10]:


plt.figure(figsize=(10,10))
df.boxplot()


# # Spliting of data - Training & Testing set

# In[11]:


X = np.array(df.Temperature.values)
y = np.array(df.Revenue.values)


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)


# In[14]:


len(X_test)


# # Choosing the model

# In[15]:


regressor = DecisionTreeRegressor()


# # Training the model

# In[16]:


regressor.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))


# # Testing the model

# In[17]:


y_pred = regressor.predict(X_test.reshape(-1,1))


# # Comparing the y_test with y_pred

# In[18]:


comp = pd.DataFrame({"Actual Values":y_test.reshape(-1),
                     "Predicted Values":y_pred.reshape(-1)})


# In[19]:


comp


# In[20]:


plt.scatter(X_test,y_test, color = 'red')
plt.scatter(X_test,y_pred, color = 'green')
plt.xlabel('X_test')
plt.ylabel('y_test/y_pred')


# In[21]:


sns.heatmap(comp.corr(), annot=True, cmap='Greens')


# In[22]:


plt.figure(figsize=(10,10))
comp.boxplot()


# # Performance

# In[23]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# In[24]:


r2_score(y_test,y_pred)


# In[25]:


mean_squared_error(y_test,y_pred)


# In[26]:


mean_absolute_error(y_test,y_pred)

