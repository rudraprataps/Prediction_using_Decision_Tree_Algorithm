#!/usr/bin/env python
# coding: utf-8

# # Name:- Rudra Pratap Singh
# 
# # Task 4:- Prediction_using_Decision_Tree_Algorithm
# 
# # The Sparks Foundation
# 
# # Iot & Computer Vision Intern

# In[26]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sbn
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics


# In[27]:


dataset = pd.read_csv("F:\Projects\Prediction_using_Decision_Tree_Algorithm\Iris.csv") #Your folder location please check from properties
dataset.head() #this function helps us to view the dataset


# In[28]:


dataset.describe() #to view the statistical data of the iris dataset


# In[29]:


dataset.info() #prints a concise summary of the dataset


# In[30]:


dataset.shape


# In[31]:


dataset.isna().sum()


# In[32]:


dataset.duplicated().sum()


# In[33]:


sbn.FacetGrid(dataset,hue="Species").map(plt.scatter,'SepalLengthCm','SepalWidthCm').add_legend()
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.show()


# In[34]:


sbn.FacetGrid(dataset,hue="Species").map(plt.scatter,'PetalLengthCm','PetalWidthCm').add_legend()
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.show()


# In[35]:


dataset['Species'].value_counts().plot(kind = 'pie', autopct = '%1.1f%%', shadow = True, explode = [0.08,0.08,0.08])


# In[36]:


x = dataset.iloc[:, 1:-1].values #x is the matrix of features
y = dataset.iloc[:,-1].values #y is a vector of observed outcomes


# In[37]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=0)
#we would have 30 observation in test set and 120 observations in the training set


# In[38]:


classifier = DecisionTreeClassifier(criterion = 'entropy')
classifier.fit(x_train,y_train)


# In[39]:


plt.figure(figsize=(15,10))
tree.plot_tree(classifier, filled = True)


# # Thank you So much, please do give your comments. Thank you 
