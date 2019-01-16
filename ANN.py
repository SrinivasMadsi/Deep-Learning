# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 16:34:14 2019

@author: srinivas.madsi
"""
# Churn Prediction in Bank using ANN

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Importing dataaset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values
# Treating Categorical variables of Gender and Country
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1]) # Gender

labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2]) # Country
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:] # Avoiding Dummy Variable Trap

# Splitting Dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Lets make ANN
# Importing necessary Keras library
import keras
from keras.models import Sequential # To initialize neural network
from keras.layers import Dense # To use layers
from keras.layers import Dropout

# Initialising ANN
classifier = Sequential()

# Adding input layer and first hidden layer
# There are 11 independent and 1 dependent variables, so avg=11+1/2 =6 layers
# init='uniform' --> It  adds weights
#input_dim=11,Since at the starting it doesn't know what are the inputs. So we pass independent variables
# Use RELU activation function for hidden layers and Sigmoid for output
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
classifier.add(Dropout(p=0.1)) # use to reduce overfitting by disabling neurons 10 percent

# Adding second hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
# Adding Output layer
# Choose softmax function  instead of 'sigmoid' if your output having more than 2 categories
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

# Compiling ANN
# Adam = StochasticGradientDescent Algo, loss= Loss function here for 2 categories
#If more than 3 categories then loss function =categorical_crossentrophy
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Fitting the ANN to the Training Set
classifier.fit(X_train,y_train,batch_size=10,nb_epoch=100)

# Making the predictions and evaluating the model
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

# Evaluation, Improving and Tuning the ANN
#K-Fold is in scikit but we use keras here so use below library
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential # To initialize neural network
from keras.layers import Dense # To use layers

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier #Here classifier is local
# Declare classifier again to make it global
classifier = KerasClassifier(build_fn = build_classifier,batch_size=10,nb_epoch=100)
accuracies = cross_val_score(estimator = classifier,X = X_train, y = y_train, cv = 10,n_jobs = -1)

mean = accuracies.mean()
variance = accuracies.std()

# Improve the ANN
# Dropout regularization to reduce overfitting if needed
from keras.layers import Dropout
#Then run above code from implementation of ANN by disabling 10%

#Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential # To initialize neural network
from keras.layers import Dense # To use layers

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
    classifier.compile(optimizer= optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier #Here classifier is local
# Declare classifier again to make it global
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size':[25,32],
              'nb_epoch':[100,500],
              'optimizer':['adam','rmsprop']}
grid_search = GridSearchCV(estimator = classifier,param_grid=parameters,
                           scoring='accuracy',cv=10)

grid_search = grid_search.fit(X_train,y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


