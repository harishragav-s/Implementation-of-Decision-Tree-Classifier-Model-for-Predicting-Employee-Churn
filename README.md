# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

### Algorithm

1. Load the employee dataset and display basic information, including null values and class distribution of the left column.
2. Encode the categorical salary column using Label Encoding.
3. Define the features (X) and target (y) by selecting relevant columns.
4. Split the data into training and testing sets (80-20 split).
5. Initialize a Decision Tree Classifier with the entropy criterion and train it on the training data.
6. Predict the target values for the test set.
7. Calculate and display the model's accuracy.
8. Compute and display the confusion matrix for the predictions.
9. Predict the left status for a new employee sample.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Harish Ragav S
RegisterNumber: 212222110013

import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
print(data.head())
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

*/
```

## Output:

![Screenshot 2024-10-18 143253](https://github.com/user-attachments/assets/64f2cd29-75fd-4b7a-9890-3f98c620bc7a)

![Screenshot 2024-10-18 143259](https://github.com/user-attachments/assets/241911d5-7b91-446a-bd79-fd762e0f3b3c)

![Screenshot 2024-10-18 143307](https://github.com/user-attachments/assets/24dfd526-82bd-4564-918a-47d005ac357d)

![Screenshot 2024-10-18 143313](https://github.com/user-attachments/assets/2435dd99-6c52-4bee-9ea6-98bc20551f56)

## Accuracy:
![Screenshot 2024-10-18 143327](https://github.com/user-attachments/assets/2d201ebd-7425-4361-bfaf-c19901280702)




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
