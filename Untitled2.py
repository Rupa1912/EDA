#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


train = pd.read_csv('titanic.csv')


# In[3]:


train.head()


# In[4]:


train.isnull()


# In[5]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[6]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train)


# In[7]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train)


# In[8]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# In[9]:


train['Age'].hist(bins=30,color='darkred',alpha=0.3)


# In[10]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# In[11]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


# In[14]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# In[13]:


train.drop('Cabin',axis=1,inplace=True)


# In[15]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[16]:


train.head()


# In[17]:


train.dropna(inplace=True)


# In[18]:


pd.get_dummies(train['Embarked'],drop_first=True).head()


# In[19]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[20]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[21]:


train.head()


# In[22]:


train = pd.concat([train,sex,embark],axis=1)


# In[23]:


train.head()


# In[24]:


train.drop('Survived',axis=1).head()


# In[25]:


train['Survived'].head()


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)


# In[28]:


from sklearn.linear_model import LogisticRegression


# In[29]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[30]:


predictions = logmodel.predict(X_test)


# In[31]:


from sklearn.metrics import confusion_matrix


# In[32]:


accuracy=confusion_matrix(y_test,predictions)


# In[33]:


accuracy


# In[34]:


from sklearn.metrics import accuracy_score


# In[35]:


accuracy=accuracy_score(y_test,predictions)
accuracy


# In[36]:


predictions


# In[37]:


from sklearn.metrics import classification_report


# In[38]:


print(classification_report(y_test,predictions))


# In[ ]:




