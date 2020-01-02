"""
Linear Regression fits a linear model with coefficients  𝜃=(𝜃1,...,𝜃𝑛)  to minimize the 'residual sum
of squares' between the independent x in the dataset, and the dependent y by the linear approximation.
"""

"""
Modeling
Using sklearn package to model data.
"""

from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

"""
Plot outputs
we can plot the fit line over the data:
'''
'''
plt.scatter(train.x, train.y,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel(str)
plt.ylabel(str)
"""

'''
Evaluation
we compare the actual values and predicted values to calculate the accuracy of a regression model. 
Evaluation metrics provide a key role in the development of a model, as it provides insight to areas that require improvement.

There are different model evaluation metrics, lets use MSE here to calculate the accuracy of our model based on the test set:

Mean absolute error: It is the mean of the absolute value of the errors. 
This is the easiest of the metrics to understand since it’s just average error.

Mean Squared Error (MSE): Mean Squared Error (MSE) is the mean of the squared error. 
It’s more popular than Mean absolute error because the focus is geared more towards large errors. 
This is due to the squared term exponentially increasing larger errors in comparison to smaller ones.
MSE is used to describe the error between the actual value with the predict value, the smaller of the MSE the more precise
of the predicted value. In a word we need a MSE as small as enough.

Root Mean Squared Error (RMSE): This is the square root of the Mean Square Error.

R-squared is not error, but is a popular metric for accuracy of your model. 
It represents how close the data are to the fitted regression line. The higher the R-squared, 
the better the model fits your data. 
Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
'''

from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )