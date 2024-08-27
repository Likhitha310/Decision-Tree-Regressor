#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


# In[2]:


df = pd.read_csv("creditcard.csv")


# # Data Analysis

# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.columns


# In[8]:


df.Class.value_counts()


# In[9]:


plt.figure(figsize=(50,50))
sns.heatmap(df.corr(), annot=True, cmap='Greens')


# # Feature Importance/Selection

# In[10]:


X = df.iloc[:,:-1]
y = df.iloc[:,-1]


# In[11]:


model = ExtraTreesClassifier()


# In[12]:


model.fit(X,y)


# In[13]:


model.feature_importances_


# In[14]:


plt.figure(figsize=(10,10))
feat = pd.Series(model.feature_importances_, index=X.columns)
feat.nlargest(18).plot(kind='barh')
plt.grid()


# In[15]:


plots = feat.nlargest(18)


# In[16]:


plots.index


# In[17]:


cols = ['V17', 'V14', 'V12', 'V10', 'V11', 'V16', 'V18', 'V9', 'V4', 'V3', 'V7',
       'V21', 'V1', 'V26', 'Time', 'V2', 'V19', 'V8']
X_new = X[cols]


# In[18]:


X_new.head(1)


# In[19]:


X.shape


# In[20]:


X_new.shape


# # Spliting the data into sets

# In[21]:


skf = StratifiedKFold(n_splits=10)


# In[22]:


for train_index, test_index in skf.split(X,y):
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  y_train, y_test = y.iloc[train_index], y.iloc[test_index]


# In[23]:


for train_index, test_index in skf.split(X_new,y):
  X_new_train, X_new_test = X_new.iloc[train_index], X_new.iloc[test_index]
  y_new_train, y_new_test = y.iloc[train_index], y.iloc[test_index]


# In[24]:


X_train.shape


# In[25]:


X_new_train.shape


# # Model Selection

# In[26]:


decision = DecisionTreeClassifier()
randomf = RandomForestClassifier()


# # Hyper Parameter Tuning for RandomForestClassifier

# In[27]:


n_estimators = [int(i) for i in np.linspace(100,1200,12)]
max_features = ['auto', 'sqrt']
max_depth = [int(i) for i in np.linspace(5,30,5)]
min_samples_split = [2,5,10,15,100]
min_samples_leaf = [1,2,5,10]


# In[28]:


parameters = {
    'n_estimators':n_estimators,
    'max_features':max_features,
    'max_depth':max_depth,
    'min_samples_split':min_samples_split,
    'min_samples_leaf':min_samples_leaf
}


# In[29]:


parameters


# In[30]:


rf_model = RandomizedSearchCV(estimator=randomf,
                              param_distributions=parameters,
                              scoring='neg_mean_squared_error',
                              n_jobs=1,
                              cv=5,
                              verbose=2,
                              random_state=42
                              )


# In[ ]:


rf_model.fit(X_train,y_train)


# In[ ]:


randomf.fit(X_train,y_train)


# In[ ]:


y_pred = randomf.predict(X_test)


# In[ ]:


accuracy_score(y_test,y_pred)

