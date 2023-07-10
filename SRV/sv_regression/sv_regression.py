import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

dataset = pd.read_csv('Position_salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# print(x)
# print(y)
print("-----")

# transform y into 2d array from a standard array
y = y.reshape(len(y), 1)
# print(y)
print("-----")

# Adding feature scaling for SVR: Both x and y
from sklearn.preprocessing import StandardScaler
svr_fs_x = StandardScaler()
svr_fs_y = StandardScaler()
x = svr_fs_x.fit_transform(x)
y = svr_fs_y.fit_transform(y)

# print(x)
# print(y)

print("-----")

# train SVR
from sklearn.svm import SVR

regressor = SVR(kernel= 'rbf')
regressor.fit(x, y)

y_pred = svr_fs_y.inverse_transform(regressor.predict(svr_fs_x.transform([[6.5]])).reshape(-1, 1))
# y_pred = regressor.predict(svr_fs_x.transform([[6.5]])).reshape(-1, 1)
# y_pred = y_pred.ravel()
print(y_pred)

plt.scatter(svr_fs_x.inverse_transform(x), svr_fs_y.inverse_transform(y), color = "red")
plt.plot(svr_fs_x.inverse_transform(x), svr_fs_y.inverse_transform(regressor.predict(x).reshape(-1, 1)), color = 'blue')
plt.title("Predicted Salaries (SVR)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()