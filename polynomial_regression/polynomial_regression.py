import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
# Train the model

# build linear regression module...
from sklearn.linear_model import LinearRegression

lin_regressor = LinearRegression()
lin_regressor.fit(x,y)

y_pred = lin_regressor.predict(x)

# print(y_pred)


# build linear pylinomial regression module...
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree= 4)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)

plt.scatter(x, y, color = "red")
plt.plot(x, lin_regressor.predict(x))
plt.title("Predicted Salaries")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


plt.scatter(x, y, color = "red")
plt.plot(x, lin_reg_2.predict(x_poly))
plt.title("Predicted Salaries")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

temp = lin_regressor.predict([[6.5]])

temp2 = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))

print(temp, temp2)

