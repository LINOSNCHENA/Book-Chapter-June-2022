import numpy as np
from sklearn.linear_model import LinearRegression
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])

model = LinearRegression()
model.fit(x, y)
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print(f"1-coefficient of determination: {r_sq}")
# coefficient of determination: 0.7158756137479542

print(f"2-intercept: {model.intercept_}")
# intercept: 5.633333333333329

print(f"slope: {model.coef_}")
# slope: [0.54]

new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
print(f"3-intercept: {new_model.intercept_}")
#intercept: [5.63333333]

print(f"slope: {new_model.coef_}")
# slope: [[0.54]]

y_pred = model.predict(x)
print(f"4-predicted response:\n{y_pred}")
# predicted response:
# [ 8.33333333 13.73333333 19.13333333 24.53333333 29.93333333 35.33333333]

y_pred = model.intercept_ + model.coef_ * x
print(f"5-predicted response:\n{y_pred}")

x_new = np.arange(5).reshape((-1, 1))
x_new
# array([[0],[1],[2], [3],[4]])
y_new = model.predict(x_new)
y_new
print('6-Answer-',y_new)
# array([5.63333333, 6.17333333, 6.71333333, 7.25333333, 7.79333333])
# https://realpython.com/linear-regression-in-python/
