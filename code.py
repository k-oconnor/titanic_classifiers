from ast import increment_lineno
from pickle import TRUE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error,accuracy_score,roc_auc_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
import re

valid = pd.read_csv("test.csv")
base = pd.read_csv("train.csv")



def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
            return title_search.group(1)
    return ""

base.loc[base.Sex != 'male', 'Sex'] = 1
base.loc[base.Sex == 'male', 'Sex'] = 0

base['Has_Cabin'] = base["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

base.Cabin.fillna('0', inplace=True)
base.loc[base.Cabin.str[0] == 'A', 'Cabin'] = 1
base.loc[base.Cabin.str[0] == 'B', 'Cabin'] = 2
base.loc[base.Cabin.str[0] == 'C', 'Cabin'] = 3
base.loc[base.Cabin.str[0] == 'D', 'Cabin'] = 4
base.loc[base.Cabin.str[0] == 'E', 'Cabin'] = 5
base.loc[base.Cabin.str[0] == 'F', 'Cabin'] = 6
base.loc[base.Cabin.str[0] == 'G', 'Cabin'] = 7
base.loc[base.Cabin.str[0] == 'T', 'Cabin'] = 8
base['Cabin'] = base['Cabin'].astype(int)


base.Fare.fillna(1, inplace=True)
base.loc[ base['Fare'] <= 7.91, 'Fare'] = 0
base.loc[(base['Fare'] > 7.91) & (base['Fare'] <= 14.454), 'Fare'] = 1
base.loc[(base['Fare'] > 14.454) & (base['Fare'] <= 31), 'Fare'] = 2
base.loc[ base['Fare'] > 31, 'Fare'] = 3
base['Fare'] = base['Fare'].astype(int)

base['Title'] = base['Name'].apply(get_title)
    # Group all non-common titles into one single grouping "Rare"
mapping = {'Mlle': 'Rare', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Rare', 'Rev': 'Mr',
               'Don': 'Mr', 'Mme': 'Rare', 'Jonkheer': 'Mr', 'Lady': 'Mrs',
               'Capt': 'Mr', 'Countess': 'Rare', 'Ms': 'Miss', 'Dona': 'Rare'}
base.replace({'Title': mapping}, inplace=True)
    # Mapping titles
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
base['Title'] = base['Title'].map(title_mapping)
base['Title'] = base['Title'].fillna(0)

for col in base.columns:
    if base[col].isnull().mean()*100>40:
        base.drop(col,axis=1,inplace=True)

f = lambda x: x.median() if np.issubdtype(x.dtype, np.number) else x.mode().iloc[0]
base = base.fillna(base.groupby('SibSp').transform(f))

le=LabelEncoder()
for col in base.columns:
    if base[col].dtypes == object:
        base[col]= le.fit_transform(base[col])

y = base['Survived']
X = base[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Cabin','Has_Cabin','Title']]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

def train_models(X_train, y_train):
    
    #use Decision Tree

    tree = DecisionTreeClassifier(max_depth = 4, random_state = 0)
    tree.fit(X_train, y_train)
    y_pred_tree = tree.predict(X_test)
  
    #use the RandomForest classifier

    rf = RandomForestClassifier(n_estimators = 100,max_features =75, random_state = 0)
    rf.fit(X_train, y_train)
    y_pred_rf= rf.predict(X_test)

    #from sklearn.svm import SVC
    svr= SVC(kernel = 'rbf')
    svr.fit(X_train, y_train)
    y_pred_svr = svr.predict(X_test)
    
    #from sklearn.svm import SVC
    svr_l= SVC(kernel = 'linear')
    svr_l.fit(X_train, y_train)
    y_pred_svr_linear = svr_l.predict(X_test)

    # use the knn classifier
    knn = neighbors.KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)

    # Using gradient boosted machine
    gbm = GradientBoostingClassifier(min_samples_split=2,max_depth=3)
    gbm.fit(X_train,y_train)
    y_pred_gbm = gbm.predict(X_test)
    
    # Using a logistic regression
    logit = LogisticRegression(penalty='l1', solver='liblinear')
    logit.fit(X_train,y_train)
    y_pred_logit = logit.predict(X_test)
    AUC_logit = roc_auc_score(y_test, y_pred_logit)

  # metrics of decision tree classifier
    meanAbErr_tree= metrics.mean_absolute_error(y_test, y_pred_tree)
    meanSqErr_tree= metrics.mean_squared_error(y_test, y_pred_tree)
    rootMeanSqErr_tree= np.sqrt(metrics.mean_squared_error(y_test, y_pred_tree))
    AUC_tree = roc_auc_score(y_test, y_pred_tree)

  # metrics of random forest classifier
    meanAbErr_rf= metrics.mean_absolute_error(y_test, y_pred_rf)
    meanSqErr_rf= metrics.mean_squared_error(y_test, y_pred_rf)
    rootMeanSqErr_rf= np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf))
    AUC_rf = roc_auc_score(y_test, y_pred_rf)

    # metrics of knn classifier
    meanAbErr_knn = metrics.mean_absolute_error(y_test, y_pred_knn)
    meanSqErr_knn = metrics.mean_squared_error(y_test, y_pred_knn)
    rootMeanSqErr_knn= np.sqrt(metrics.mean_squared_error(y_test, y_pred_knn))
    AUC_knn = roc_auc_score(y_test, y_pred_knn) 

  # metrics of svc
    meanAbErr_svr = metrics.mean_absolute_error(y_test, y_pred_svr_linear)
    meanSqErr_svr = metrics.mean_squared_error(y_test, y_pred_svr_linear)
    rootMeanSqErr_svr= np.sqrt(metrics.mean_squared_error(y_test, y_pred_svr_linear))
    AUC_svr = roc_auc_score(y_test, y_pred_svr_linear) 

  # metrics of gbm
    meanAbErr_gbm = metrics.mean_absolute_error(y_test, y_pred_svr_linear)
    meanSqErr_gbm = metrics.mean_squared_error(y_test, y_pred_svr_linear)
    rootMeanSqErr_gbm= np.sqrt(metrics.mean_squared_error(y_test, y_pred_svr_linear))
    AUC_gbm = roc_auc_score(y_test, y_pred_gbm) 

  # metrics of logit
    meanAbErr_logit = metrics.mean_absolute_error(y_test, y_pred_logit)
    meanSqErr_logit = metrics.mean_squared_error(y_test, y_pred_logit)
    rootMeanSqErr_logit= np.sqrt(metrics.mean_squared_error(y_test, y_pred_logit))
    AUC_logit = roc_auc_score(y_test, y_pred_logit) 

  #print the training accurancy of each model:
    print('[1]Decision Tree Training Accurancy: ', accuracy_score(y_test,y_pred_tree))
    print('Mean Absolute Error:', meanAbErr_tree)
    print('Mean Square Error:', meanSqErr_tree)
    print('Root Mean Square Error:', rootMeanSqErr_tree)
    print('AUC:', AUC_tree)
    print('\t')
    print('[2]RandomForestRegressor Training Accurancy: ',accuracy_score(y_test,y_pred_rf))
    print('Mean Absolute Error:', meanAbErr_rf)
    print('Mean Square Error:', meanSqErr_rf)
    print('Root Mean Square Error:', rootMeanSqErr_rf)
    print('AUC:', AUC_rf)
    print('\t')    
    print('[3]SupportvectorRegression Accuracy(rbf): ', accuracy_score(y_test,y_pred_svr))
    print('\t')
    print('[4]SupportvectorRegression Accuracy(linear): ', accuracy_score(y_test,y_pred_svr_linear))
    print('Mean Absolute Error:', meanAbErr_svr)
    print('Mean Square Error:', meanSqErr_svr)
    print('Root Mean Square Error:', rootMeanSqErr_svr)
    print('AUC:', AUC_svr)
    print('\t')
    print('[5]knn Training Accurancy: ', accuracy_score(y_test,y_pred_knn))
    print('Mean Absolute Error:', meanAbErr_knn)
    print('Mean Square Error:', meanSqErr_knn)
    print('Root Mean Square Error:', rootMeanSqErr_knn)
    print('AUC:', AUC_knn)
    print('\t')
    print('[6]gbm Training Accurancy: ', accuracy_score(y_test,y_pred_gbm))
    print('Mean Absolute Error:', meanAbErr_gbm)
    print('Mean Square Error:', meanSqErr_gbm)
    print('Root Mean Square Error:', rootMeanSqErr_gbm)
    print('AUC:', AUC_gbm)
    print('\t')
    print('[6]logit Training Accurancy: ', accuracy_score(y_test,y_pred_logit))
    print('Mean Absolute Error:', meanAbErr_logit)
    print('Mean Square Error:', meanSqErr_logit)
    print('Root Mean Square Error:', rootMeanSqErr_logit)
    print('AUC:', AUC_logit)
    print('\t')
    
train_models(X_train,y_train)

# Validation and prediction

valid = pd.read_csv("test.csv")

valid.loc[valid.Sex != 'male', 'Sex'] = 1
valid.loc[valid.Sex == 'male', 'Sex'] = 0

valid['Has_Cabin'] = valid["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

valid.Cabin.fillna('0', inplace=True)
valid.loc[valid.Cabin.str[0] == 'A', 'Cabin'] = 1
valid.loc[valid.Cabin.str[0] == 'B', 'Cabin'] = 2
valid.loc[valid.Cabin.str[0] == 'C', 'Cabin'] = 3
valid.loc[valid.Cabin.str[0] == 'D', 'Cabin'] = 4
valid.loc[valid.Cabin.str[0] == 'E', 'Cabin'] = 5
valid.loc[valid.Cabin.str[0] == 'F', 'Cabin'] = 6
valid.loc[valid.Cabin.str[0] == 'G', 'Cabin'] = 7
valid.loc[valid.Cabin.str[0] == 'T', 'Cabin'] = 8
valid['Cabin'] = valid['Cabin'].astype(int)

valid.Fare.fillna(1, inplace=True)
valid.loc[valid['Fare'] <= 7.91, 'Fare'] = 0
valid.loc[(valid['Fare'] > 7.91) & (valid['Fare'] <= 14.454), 'Fare'] = 1
valid.loc[(valid['Fare'] > 14.454) & (valid['Fare'] <= 31), 'Fare'] = 2
valid.loc[valid['Fare'] > 31, 'Fare'] = 3
valid['Fare'] = valid['Fare'].astype(int)

valid['Title'] = valid['Name'].apply(get_title)
    # Group all non-common titles into one single grouping "Rare"
mapping = {'Mlle': 'Rare', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Rare', 'Rev': 'Mr',
'Don': 'Mr', 'Mme': 'Rare', 'Jonkheer': 'Mr', 'Lady': 'Mrs',
'Capt': 'Mr', 'Countess': 'Rare', 'Ms': 'Miss', 'Dona': 'Rare'}
valid.replace({'Title': mapping}, inplace=True)
    # Mapping titles
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
valid['Title'] = valid['Title'].map(title_mapping)
valid['Title'] = valid['Title'].fillna(0)

for col in valid.columns:
    if valid[col].isnull().mean()*100>40:
        valid.drop(col,axis=1,inplace=True)

f = lambda x: x.median() if np.issubdtype(x.dtype, np.number) else x.mode().iloc[0]
valid = valid.fillna(valid.groupby('SibSp').transform(f))

le=LabelEncoder()
for col in valid.columns:
    if valid[col].dtypes == object:
        valid[col]= le.fit_transform(valid[col])

X_valid = valid[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Cabin','Has_Cabin','Title']]
# rf = RandomForestClassifier(n_estimators = 100,max_features =75, random_state = 0)
# rf.fit(X_train, y_train)
# valid['Survived'] = rf.predict(X_valid)

# tree = DecisionTreeClassifier(max_depth = 4, random_state = 0)
# tree.fit(X_train, y_train)
# valid['Survived'] = tree.predict(X_valid)

gbm = GradientBoostingClassifier(max_depth=3)
gbm.fit(X_train,y_train)
valid['Survived'] = gbm.predict(X_valid)


results = valid[["PassengerId", "Survived"]]
results.to_csv("results.csv",header=True,index=False)