
# coding: utf-8

# In[321]:


# used libraries
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


# In[322]:


# import the dataset
dt = pd.read_csv("dataset.csv")
dt.shape


# In[323]:


# separate data (X and Y)
array = dt.values
X = array[:, 0:216]
y = array[:, 216]
validation_size = 0.30
seed = 7
X.shape


# In[324]:


X_train, X_validation, y_train, y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
X_train.shape


# In[325]:


# L1-based feature selection
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train, y_train)
modellsvc = SelectFromModel(lsvc, prefit=True)
X_train_new = modellsvc.transform(X_train)
X_train_new.shape


# In[326]:


# test six classifiers
scoring = 'accuracy'
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train_new, y_train, cv=kfold, scoring=scoring)
	results.append(cv_results.mean())
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


# In[327]:


X_validation_new = modellsvc.transform(X_validation)
X_validation_new.shape


# In[328]:


best_model_idx = results.index(max(results))
best_model = models[best_model_idx]
best_model[0]


# In[329]:


best_model[1].fit(X_train_new, y_train)
predictions = best_model[1].predict(X_validation_new)
accuracy_score(y_validation, predictions)

