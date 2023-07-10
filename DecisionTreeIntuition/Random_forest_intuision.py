import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# print(x)
# print(y)

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators= 10, random_state = 0)
regressor.fit(x, y)
              
y_pred = regressor.predict(x)
print(y_pred)

x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = "red")
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title("Predicted Salaries (Random forest intuision)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()