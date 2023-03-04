import pandas as pd
from sklearn.datasets import load_digits

#Creating load_digits object
digits = load_digits()
#Looking for features of dataset
print(dir(digits))
print(digits.target_names)

#Creating dataframe
df = pd.DataFrame(digits.data,columns=digits.feature_names)
print(df.head())

#Creating target column in dataframe
df["target"] = digits.target
print(df.head())

#Creeating model
X = df.drop(["target"],axis="columns")
y = df.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))





