# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Collect a labeled dataset of emails, distinguishing between spam and non-spam.

2.Preprocess the email data by removing unnecessary characters, converting to lowercase, removing stop words, and performing stemming or lemmatization.

3.Extract features from the preprocessed text using techniques like Bag-of-Words or TF-IDF.

4.Split the dataset into a training set and a test set.

5.Train an SVM model using the training set, selecting the appropriate kernel function and hyperparameters.

6.Evaluate the trained model using the test set, considering metrics such as accuracy, precision, recall, and F1 score.

7.Optimize the model's performance by tuning its hyperparameters through techniques like grid search or random search.

8.Deploy the trained and fine-tuned model for real-world use, integrating it into an email server or application.

9.Monitor the model's performance and periodically update it with new data or adjust hyperparameters as needed

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: BALASUDHAN P
RegisterNumber: 212222240017


import chardet
file="/content/spam.csv"
with open(file,"rb") as rawdata:
  result=chardet.detect(rawdata.read(10000))
result

import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding="Windows-1252")

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

*/
```

## Output:

![image](https://github.com/BALASUDHAN18/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118807740/bd7628bf-5459-4b0c-8fde-34f62589d9ea)

![image](https://github.com/BALASUDHAN18/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118807740/c30da0a3-ec6a-48f8-927b-69cd2c891065)

![image](https://github.com/BALASUDHAN18/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118807740/4afc5e53-0b8b-420f-8aee-1434e9edd157)

![image](https://github.com/BALASUDHAN18/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118807740/04f82090-4c20-4422-873a-57eed6feaca1)

![image](https://github.com/BALASUDHAN18/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118807740/5ad37680-e95d-4982-ae4d-e046f8baa4ae)



![image](https://github.com/BALASUDHAN18/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118807740/74d31b57-8a96-4baf-a7c7-c5439ca2c9b1)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
