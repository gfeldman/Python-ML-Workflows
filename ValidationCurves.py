#
# Parameter Tuning - Validation Curves
#

import pandas as pd
import numpy as np

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',header=None)


# Get the data ready for Scikit-learn:
# 1. put the input variables into a numpy array called X 
# 2. put the label vectors into a vector called y
# 3. convert the string labels to numbers using LabelEncoder
# 4. create training and test sets using train_test_split
#
from sklearn.preprocessing import LabelEncoder
X = df.loc[:,2:].values
y = df.loc[:,1].values

le = LabelEncoder()
y = le.fit_transform(y)


from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state = 47906)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# rather than use PCA, we try l2 regularization

pipe_lr = Pipeline([('scl',StandardScaler()),('clf',LogisticRegression(penalty='l2',random_state=0))])

# The validation curves plot training and test accuracy against different values of hyper parameters 
#
# scikit learn will generate validation curves for the pipeline and automate the cross validation process
#
import matplotlib.pyplot as plt
from sklearn.learning_curve import validation_curve

#parameter range we're going to search over 
param_range = [0.001,0.01,1.0,10.0, 100.0]

train_scores, test_scores = validation_curve(estimator=pipe_lr,X=X_train,y=y_train,param_name='clf__C',param_range = param_range, cv=10)

train_mean = np.mean(train_scores,axis=1)
train_std = np.std(train_scores,axis=1)
test_mean = np.mean(test_scores,axis=1)
test_std = np.std(test_scores,axis=1)

plt.plot(param_range,train_mean,color='blue',marker='o',markersize=5,label="Training Accuracy")
plt.fill_between(param_range,train_mean +train_std,train_mean - train_std,alpha=0.15,color='blue')
plt.plot(param_range,test_mean, color='green',linestyle='--',marker='s',markersize=5,label="Validation Accuracy")
plt.fill_between(param_range,test_mean +test_std, test_mean - test_std,alpha=0.15,color='green')
plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.8,1.0])
plt.show()