import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("data/placementdata.csv")
print(df.head(5))
print(df.tail(5))
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df['PlacementStatus'].value_counts())
# print(df.corr(numeric_only=True))
# plt.figure(figsize=(10,8))
# sns.heatmap(
#     df.corr(numeric_only=True),
#     annot=True
# )
# plt.show()

# df.hist(figsize=(10,12))
# plt.show()

# sns.boxplot(df['CGPA'])
# plt.show()

# sns.boxplot(
#     x='PlacementStatus',
#     y='CGPA',
#     data=df
# )
# plt.show()

print(df.isnull().sum())
print("Before Dropping" ,df.duplicated().sum())
print(df.dtypes)
df.drop_duplicates(inplace=True)
print("After dropping",df.duplicated().sum())

# object data
print(df['PlacementStatus'].unique())
print(df['ExtracurricularActivities'].unique())
print(df['PlacementTraining'].unique())
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
df['PlacementStatus']=encoder.fit_transform(df['PlacementStatus'])
df['ExtracurricularActivities']=encoder.fit_transform(df['ExtracurricularActivities'])
df['PlacementTraining']=encoder.fit_transform(df['PlacementTraining'])

print(df.head(5))
print(df.dtypes)
x=df.drop("PlacementStatus",axis=1)
y=df['PlacementStatus']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)
print(len(x_test))
print(len(x_train))
print(len(y_test))
print(len(x_train))

# Logistic Regression

# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# model=LogisticRegression(max_iter=100)
# model.fit(x_train,y_train)
# predict=model.predict(x_test)
# accuracy=accuracy_score(y_test,predict)
# print("Accuracy by using the Logistic Regression is : ",accuracy)
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix

# print(classification_report(y_test,prediction))
# print(confusion_matrix(y_test,prediction))

# Decision tree
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
# model=DecisionTreeClassifier(max_depth=10)
# model.fit(x_train,y_train)
# prediction=model.predict(x_test)
# accuracy=accuracy_score(y_test,prediction)
# print("Accuracy by using the Decision Tree is : ",accuracy)
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix

# print(classification_report(y_test,prediction))
# print(confusion_matrix(y_test,prediction))



#Naiye bayes
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score
# model=GaussianNB()
# model.fit(x_train,y_train)
# prediction=model.predict(x_test)
# accuracy=accuracy_score(y_test,prediction)
# print("Accuracy by using the Naive bayes is :",accuracy)
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix

# print(classification_report(y_test,prediction))
# print(confusion_matrix(y_test,prediction))

# #SVM
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# model=SVC()
# model.fit(x_train,y_train)
# prediction=model.predict(x_test)
# accuracy=accuracy_score(y_test,prediction)
# print("Accuracy by using the Naive bayes is :",accuracy)
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix

# print(classification_report(y_test,prediction))
# print(confusion_matrix(y_test,prediction))

# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
model=RandomForestClassifier(n_estimators=10)
model.fit(x_train,y_train)
prediction=model.predict(x_test)
accuracy=accuracy_score(y_test,prediction)
print("Accuracy by using the Random Forest is :",accuracy)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


print(classification_report(y_test,prediction))
print(confusion_matrix(y_test,prediction))


# hyper paramter tuning in Random forest

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=10),
    param_grid,
    cv=5,
    scoring='accuracy'
)
# print("Hyperparameter tuning started...")
# grid.fit(x_train, y_train)
# print("Hyperparameter tuning completed...")
# print("Best Parameters:", grid.best_params_)
# print("Best Cross Validation Score:", grid.best_score_)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
final_model=RandomForestClassifier(n_estimators=200,random_state=10,max_depth=5)
final_model.fit(x_train,y_train)
prediction=final_model.predict(x_test)
accuracy=accuracy_score(y_test,prediction)
print("Accuracy by using the Random Forest is :",accuracy)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(classification_report(y_test, prediction))
print(confusion_matrix(y_test, prediction))

import os
import pickle

# This tells Python to step out of 'notebooks' and save it directly into 'models'


# Point directly to the models folder in your root directory
model_save_path = os.path.join("models", "placement_model.pkl")

with open(model_save_path, "wb") as f:
    pickle.dump(final_model, f)

print("Success! Clean binary model saved inside the models/ folder.")