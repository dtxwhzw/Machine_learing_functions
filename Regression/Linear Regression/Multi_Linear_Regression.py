"""
the Multiple linear regression is the extension of simple linear regession model
"""
from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])#choose multiple variable in the train set.
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)
# we'll get several coefficients
print ('Coefficients: ', regr.coef_)

"""
we've known that the parameters are the intercept and coefficients of hyperplane, sklearn can estimate them from our data.
Scikit-learn uses plain Ordinary Least Squares method to solve this problem.
"""
'''
Ordinary Least Squares (OLS)
OLS is a method for estimating the unknown parameters in a linear regression model. 
OLS chooses the parameters of a linear function of a set of explanatory variables by 
minimizing the sum of the squares of the differences between the target dependent variable 
and those predicted by the linear function. In other words,
it tries to minimizes the sum of squared errors (SSE) or mean squared error (MSE) between the 
target variable (y) and our predicted output ( 𝑦̂  ) over all samples in the dataset.

OLS can find the best parameters using of the following methods: 
- Solving the model parameters analytically using closed-form equations 
- Using an optimization algorithm (Gradient Descent, Stochastic Gradient Descent, Newton’s Method, etc.)
'''
#Prediction
y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))
"""
If  𝑦̂   is the estimated target output, y the corresponding (correct) target output, 
and Var is Variance, the square of the standard deviation, then the explained variance is estimated as follow:

𝚎𝚡𝚙𝚕𝚊𝚒𝚗𝚎𝚍𝚅𝚊𝚛𝚒𝚊𝚗𝚌𝚎(𝑦,𝑦̂ )=1−𝑉𝑎𝑟{𝑦−𝑦̂ }/𝑉𝑎𝑟{𝑦} 
The best possible score is 1.0, lower values are worse.
"""