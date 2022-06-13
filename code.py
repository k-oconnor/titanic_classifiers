from ast import increment_lineno
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,auc
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors


valid = pd.read_csv("test.csv")
base = pd.read_csv("train.csv")

f = lambda x: x.median() if np.issubdtype(x.dtype, np.number) else x.mode().iloc[0]
base = base.fillna(base.groupby('SibSp').transform(f))


le=LabelEncoder()
for col in base.columns:
    if base[col].dtypes == object:
        base[col]= le.fit_transform(base[col])

y = base['Survived']
X = base[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

for col in X.columns:
    if X[col].isnull().mean()*100>40:
        X.drop(col,axis=1,inplace=True)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

def train_models(X_train, y_train):
    
 #use Decision Tree
   
    tree = DecisionTreeRegressor(max_features=75,max_depth=4, random_state = 0)
    tree.fit(X_train, y_train)
    y_pred_tree = tree.predict(X_test)

  #use the RandomForestRegressor
    
    rf = RandomForestRegressor(n_estimators = 100,max_features =75, random_state = 0)
    rf.fit(X_train, y_train)
    y_pred_rf= rf.predict(X_test)
    
  # use the support vector regressor
    #from sklearn.svm import SVR
    svr= SVR(kernel = 'rbf')
    svr.fit(X_train, y_train)
    y_pred_svr = svr.predict(X_test)
    
    #from sklearn.svm import SVR
    svr_l= SVR(kernel = 'linear')
    svr_l.fit(X_train, y_train)
    y_pred_svr_linear = svr_l.predict(X_test)

    # use the knn regressor
    knn = neighbors.KNeighborsRegressor()
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    
  # metrics of decision tree regressor
    meanAbErr_tree= metrics.mean_absolute_error(y_test, y_pred_tree)
    meanSqErr_tree= metrics.mean_squared_error(y_test, y_pred_tree)
    rootMeanSqErr_tree= np.sqrt(metrics.mean_squared_error(y_test, y_pred_tree))

  # metrics of random forest regressor
    meanAbErr_rf= metrics.mean_absolute_error(y_test, y_pred_rf)
    meanSqErr_rf= metrics.mean_squared_error(y_test, y_pred_rf)
    rootMeanSqErr_rf= np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf))

    # metrics of knn regressor
    meanAbErr_knn = metrics.mean_absolute_error(y_test, y_pred_knn)
    meanSqErr_knn = metrics.mean_squared_error(y_test, y_pred_knn)
    rootMeanSqErr_knn= np.sqrt(metrics.mean_squared_error(y_test, y_pred_knn)) 

  # metrics of svr regressor
    meanAbErr_svr = metrics.mean_absolute_error(y_test, y_pred_svr_linear)
    meanSqErr_svr = metrics.mean_squared_error(y_test, y_pred_svr_linear)
    rootMeanSqErr_svr= np.sqrt(metrics.mean_squared_error(y_test, y_pred_svr_linear)) 

  #print the training accurancy of each model:

    print('[1]Decision Tree Training Accurancy: ', r2_score(y_test,y_pred_tree))
    print('Mean Absolute Error:', meanAbErr_tree)
    print('Mean Square Error:', meanSqErr_tree)
    print('Root Mean Square Error:', rootMeanSqErr_tree)
    print('\t')
    print('[2]RandomForestRegressor Training Accurancy: ',r2_score(y_test,y_pred_rf))
    print('Mean Absolute Error:', meanAbErr_rf)
    print('Mean Square Error:', meanSqErr_rf)
    print('Root Mean Square Error:', rootMeanSqErr_rf)
    print('\t')    
    print('[3]SupportvectorRegression Accuracy(rbf): ', r2_score(y_test,y_pred_svr))
    print('\t')
    print('[4]SupportvectorRegression Accuracy(linear): ', r2_score(y_test,y_pred_svr_linear))
    print('Mean Absolute Error:', meanAbErr_svr)
    print('Mean Square Error:', meanSqErr_svr)
    print('Root Mean Square Error:', rootMeanSqErr_svr)
    print('\t')
    print('[5]knn Training Accurancy: ', r2_score(y_test,y_pred_knn))
    print('Mean Absolute Error:', meanAbErr_knn)
    print('Mean Square Error:', meanSqErr_knn)
    print('Root Mean Square Error:', rootMeanSqErr_knn)
    print('\t')

train_models(X_train,y_train)