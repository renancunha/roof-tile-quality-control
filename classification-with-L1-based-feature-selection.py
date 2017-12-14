
# coding: utf-8

# Import libraries:

# In[67]:


import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# ### Dataset
# This dataset contains audio features (36 features) extracted from 280 samples. For each feature, we have 6 statistics (mean, median, std, std by mean, max and min.
# 
# So, we expect a 280x(216+1) matrix, where the last column is the "label" (1 = good roof-tile or 2 = roof-tile with problems).

# In[68]:


dt = pd.read_csv("dataset.csv")
dt.shape


# Lets see how our data looks like:

# In[69]:


dt.head()


# Lets separate the data:
# 
# X will be a matrix containing all features from all samples.
# 
# Y will be a vector containing the labels of all observations.

# In[70]:


array = dt.values
X = array[:, 0:216]
y = array[:, 216]
X.shape


# Now, we need to make the data sets to train and to validate the models. The choosed proportion is: 70% to test and 30% to validate.

# In[71]:


validation_size = 0.30
seed = 7
X_train, X_validation, y_train, y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
X_train.shape


# ### L1-based feature selection
# Our dataset contains a lot of features (216 to be more specific).
# 
# Some features are collinear, so we can and we must to transform our data. To do that, I choosed to use a L1-based feature selection method.
# 
# Its importante to say that smaller C implies in fewer features selected.

# In[72]:


lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train, y_train)
modellsvc = SelectFromModel(lsvc, prefit=True)
X_train_new = modellsvc.transform(X_train)
X_train_new.shape


# ### Select a classifier
# We will evaluate six classifiers, to choose the best model to classify our validation data. The criteria to choose the best is the accuracy of the model on the train data.
# 
# We use a cross-validation (k-fold with k = 10) to evaluate the models

# In[73]:


models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

scoring = 'accuracy'
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train_new, y_train, cv=kfold, scoring=scoring)
	results.append(cv_results.mean())
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


# Selected the best classifier model

# In[74]:


best_model_idx = results.index(max(results))
best_model = models[best_model_idx]
best_model[0]


# Transform the validation input data to reduce the number of features
# We will use our modellsvc to do the feature selection here

# In[75]:


X_validation_new = modellsvc.transform(X_validation)
X_validation_new.shape


# ### Make the predictions on validation data
# Finally, we evaluate the accuracy of our proposed model making the predictions of X_validation_new

# In[76]:


best_model[1].fit(X_train_new, y_train)
predictions = best_model[1].predict(X_validation_new)
accuracy_score(y_validation, predictions)

