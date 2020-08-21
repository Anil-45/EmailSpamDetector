'''
change working directory to email_spam_detector(file name)
Run command python .\main\spam_detector.py
as I am using path for data from this folder in abspath
or else copy data to desired location and give that path in abspath
'''

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import os

#importing data
data=pd.read_csv(os.path.abspath("main/spam.csv"))
#splittind data 75%for training and 25% for testing
X_train,X_test,y_train,y_test=train_test_split(data["EmailText"],data["Label"],test_size=0.25,random_state=30)

#as our emails are text we are extracting features (for further details check CounterVectorizer documentation)
cv=CountVectorizer()
X_train=cv.fit_transform(X_train)
X_test=cv.transform(X_test)


#best parameters by gridsearch and placed them(if you want to tune your parameters comment this and uncomment the below code
lr=svm.SVC(C=10,gamma=0.001,kernel='linear')


'''
lr=svm.SVC()
load_param={'kernel':['linear','rbf'],'gamma':[1e-3,1e-4],'C':[1,10,100,1000]}
grid_search=GridSearchCV(lr,load_param)
grid_search.fit(X_train,y_train)
print(grid_search.best_params_)
use this to get best performance combination and place in svm.SVC()
'''
#fitting data to our model
lr.fit(X_train,y_train)
#predicting for test cases
pred=lr.predict(X_test)
#printing accuracy of prediction
print(sklearn.metrics.accuracy_score(y_test,pred))
#testing for some random input
x=["To get 2.50 pounds free call credit and details of great offers pls reply"]
#converting the text to handle
x=cv.transform(x)
#predicting with designed model
p=lr.predict(x)
print(p)

