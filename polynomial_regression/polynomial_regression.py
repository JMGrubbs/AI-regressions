import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1]
y = dataset.iloc[:, -1]
# Train the model

# build linear regression module...
from sklearn.linear_model import LinearRegression

lin_regressor = LinearRegression()
lin_regressor.fit(x,y)

print(lin_regressor.predict(100000))


# build linear pylinomial regression module...
from sklearn.preprocessing import PolynomialFeatures

