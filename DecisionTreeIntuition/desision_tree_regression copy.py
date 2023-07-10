import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

print(x)
print(y)

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x, y)
              
y_pred = regressor.predict(x)
print(y_pred)

plt.scatter(x, y, color = "red")
plt.plot(x, y_pred, color = 'blue')
plt.title("Predicted Salaries (Desison tree)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()