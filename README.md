# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary packages.
2. Read the given csv file and display the few contents of the data.Assign the features for x and y respectively.
3. Split the x and y sets into train and test sets.Convert the Alphabetical data to numeric using CountVectorizer.
4. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
5. Find the accuracy of the model.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: D.B.V. SAI GANESH
RegisterNumber:  212223240025
*/
```
```
import chardet
file = '/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result
```
```
import pandas as pd
data= pd.read_csv("/content/spam.csv",encoding='Windows-1252')
```
```
data.head()
```
```
data.info()
```
```
x=data["v1"].values
```
```
y=data["v2"].values
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
```
```
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
```
```
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
```
```
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
```
```
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:

### Result Output:
![image](https://github.com/saiganesh2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145742342/8739cec4-45a2-4a61-be3d-7dab6d7a330f)

### data.head()
![image](https://github.com/saiganesh2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145742342/6bac8890-1fa2-405d-ba94-4aeea5d376d3)

### data.info()
![image](https://github.com/saiganesh2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145742342/83d3923e-d395-4690-b0e1-ce441a6d9d07)

### data.isnull().sum()
![image](https://github.com/saiganesh2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145742342/496a86e1-2c63-4063-b076-3a60a57a72bd)

### Y_Prediction value
![image](https://github.com/saiganesh2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145742342/d4a93f8c-6f1d-43d7-b23d-c180efacfbf2)

### Accuracy Value
![image](https://github.com/saiganesh2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145742342/67fa4bb8-2171-4fe4-a8b8-04089eaddeee)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
