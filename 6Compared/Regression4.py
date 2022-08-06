import numpy as np
from sklearn.linear_model import LinearRegression

x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)
print(x)
print(y)
# array([ 4,  5, 20, 14, 32, 22, 38, 43])
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print(f"1b-coefficient of determination: {r_sq}")
# coefficient of determination: 0.8615939258756776
print(f"2b-intercept: {model.intercept_}")
# intercept: 5.52257927519819
print(f"coefficients: {model.coef_}")
# coefficients: [0.44706965 0.25502548]

y_pred = model.predict(x)
print(f"3b-predicted response:\n{y_pred}")
# predicted response:
# [ 5.77760476  8.012953   12.73867497 17.9744479  23.97529728 29.4660957
 # 38.78227633 41.27265006]

y_pred = model.intercept_ + np.sum(model.coef_ * x, axis=1)
print(f"4b-predicted response:\n{y_pred}")
# predicted response:
# [ 5.77760476  8.012953   12.73867497 17.9744479  23.97529728 29.4660957
#  38.78227633 41.27265006]

x_new = np.arange(10).reshape((-1, 2))
x_new
# array([[0, 1],
#        [2, 3],
#        [4, 5],
#        [6, 7],
#        [8, 9]])
print('5b-x_new=',x_new)
print("===================|Print-One|=========================")
y_new = model.predict(x_new)
y_new
print('6b-y_new=',y_new)
# array([ 5.77760476,  7.18179502,  8.58598528,  9.99017554, 11.3943658 ])
print("================|Print-Zambia|=========================")

# evaluate a logistic regression model using k-fold cross-validation
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
# create dataset
X, y = make_classification(n_samples=100, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# prepare the cross-validation procedure
cv = KFold(n_splits=10, random_state=1, shuffle=True)
# create model
model = LogisticRegression()
# evaluate model
print('result-kfoldx=',cv)
print('result-modelx=',model)
print('result-models=',model)
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))


# https://stackoverflow.com/questions/38015181/accuracy-score-valueerror-cant-handle-mix-of-binary-and-continuous-target