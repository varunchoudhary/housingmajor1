import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import utils
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
df= pd.read_csv("final.csv")
df.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
df.dropna(how='any')

#print(np.isnan(df))

#np.where(np.isnan(df))

income = df['SalePrice']
#print (income.head())
features = df.drop('SalePrice', axis = 1)

#--------------
from sklearn import tree
clfr = tree.DecisionTreeRegressor(max_depth=5)
clfr = clfr.fit(features,income)

X_train, X_test, y_train, y_test = train_test_split(features, income, test_size = 0.2, random_state = 0)

from sklearn.metrics import mean_squared_error
def rmse(y_test,y_pred):
      return np.sqrt(mean_squared_error(y_test,y_pred))
      
pred= clfr.predict(X_test)
print(rmse(ytest,pred)


le = preprocessing.LabelEncoder()
le.fit(income)
income=le.transform(income)






#from sklearn.cross_validation import train_test_split
print (income)
X_train, X_test, y_train, y_test = train_test_split(features, income, test_size = 0.2, random_state = 0)



from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss="hinge", penalty="l2")
predictions=(clf.fit(X_train, y_train).predict(X_test))
print(accuracy_score(y_test,predictions))


#pip install scikit-learn==0.18.rc2
#from sklearn.neural_natwork import MLPClassifier

#clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

#pred=clf.fit(features,income).predict(y_test)


clf=DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)
predictions = (clf.fit(X_train,y_train).predict(X_test))
print(accuracy_score(y_test,predictions))


clf=svm.SVC()
clf.fit(X_train,y_train)
predictions = (clf.fit(X_train,y_train).predict(X_test))
print(accuracy_score(y_test,predictions))

 


#X_train, X_test, Y_train, Y_test = train_test_split(alpha, beta, test_size=0.2,random_state=0)

