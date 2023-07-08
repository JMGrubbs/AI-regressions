import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values # x is the data we have to determine y
y = dataset.iloc[:, -1].values # y is the thing we are trying to determine..

# Fill missing data...
# No missing data 

# Changes string into numbers for the AI model...
# No strings in data set

# Changing yes and no too 1 and 0...
# Not needed for this dataset


# Creating training and test sets...
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=1)

# Fearture Scaling...
# Not needed

from sklearn.linear_model import LinearRegression
regreser = LinearRegression()
regreser.fit(X=x_train, y=y_train)

y_pred = regreser.predict(X=x_test)
# print(y_pred)

plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regreser.predict(x_train), color= "blue")
plt.title("Salary Vs exp (Training Set)")
plt.xlabel("Years of exp")
plt.ylabel("Salary")
plt.show()

plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regreser.predict(x_train), color= "blue")
plt.title("Salary Vs exp (test Set)")
plt.xlabel("Years of exp")
plt.ylabel("Salary")
plt.show()

