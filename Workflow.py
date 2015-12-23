import pandas as pd
import numpy as np

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',header=None)


# 
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


#
# Using a pipeline 
# 1. import the parts of the pipeline from scikit-learn
# 2. construct a pipeline
# 3. run fit on the pipeline
# 4. display accuracy

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

#
#	The pipeline object takes a list of tuples as input. 
#   Each tuple is of the form (id,T) where id is a unique string to identify the component and T is a transformer or estimator.
#
#
pipe_lr = Pipeline([('scl',StandardScaler()), ('pca',PCA(n_components=2)), ('clf',LogisticRegression(random_state=47906))])
pipe_lr.fit(X_train,y_train)
print('Test Accuracy: %.3f' % pipe_lr.score(X_test,y_test))



# Using Cross cross_validation

from sklearn.cross_validation import cross_val_score

scores = cross_val_score(estimator = pipe_lr,X=X_train,y=y_train,cv=10,n_jobs=1)

print('CV accuracy scores: %s' % scores)

print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores) ))