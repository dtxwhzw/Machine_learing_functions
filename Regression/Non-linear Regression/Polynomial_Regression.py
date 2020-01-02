"""
Sometimes, the trend of data is not really linear, and looks curvy. In this case we can use Polynomial regression methods.
In fact, many different regressions exist that can be used to fit whatever the dataset looks like, such as quadratic,
cubic, and so on, and it can go on and on to infinite degrees.

In essence, we can call all of these, polynomial regression, where the relationship between the independent variable x
and the dependent variable y is modeled as an nth degree polynomial in x.
Lets say you want to have a polynomial regression (let's make 2 degree polynomial):

𝑦=𝑏+𝜃1𝑥+𝜃2𝑥2
Now, the question is: how we can fit our data on this equation while we have only x values,
such as Engine Size? Well, we can create a few additional features: 1,  𝑥 , and  𝑥2 .

PloynomialFeatures() function in Scikit-learn library, drives a new feature sets from the original feature set.
That is, a matrix will be generated consisting of all polynomial combinations of the features with degree less than or
equal to the specified degree. For example, lets say the original feature set has only one feature, ENGINESIZE.
Now, if we select the degree of the polynomial to be 2, then it generates 3 features, degree=0, degree=1 and degree=2:
"""
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])


poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)
train_x_poly
"""
fit_transform takes our x values, and output a list of our data raised from power of 0 to power of 2 
(since we set the degree of our polynomial to 2).

"""

'''
 Indeed, Polynomial regression is a special case of linear regression, with the main idea of how do you select your features. 
 Just consider replacing the  𝑥  with  𝑥1 ,  𝑥21  with  𝑥2 , and so on. Then the degree 2 equation would be turn into:

𝑦=𝑏+𝜃1𝑥1+𝜃2𝑥2 
Now, we can deal with it as 'linear regression' problem. Therefore, this polynomial regression is considered to be a special 
case of traditional multiple linear regression. So, you can use the same mechanism as linear regression to solve such a problems.

so we can use LinearRegression() function to solve it:
'''
clf = linear_model.LinearRegression()
train_y_ = clf.fit(train_x_poly, train_y)
# The coefficients
print ('Coefficients: ', clf.coef_)
print ('Intercept: ',clf.intercept_)

"""
As mentioned before, Coefficient and Intercept , are the parameters of the fit curvy line. 
Given that it is a typical multiple linear regression, with 3 parameters, and knowing that the parameters are 
the intercept and coefficients of hyperplane, sklearn has estimated them from our new set of feature sets. Lets plot it:
"""
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf.intercept_[0]+ clf.coef_[0][1]*XX+ clf.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy, '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")

"""
Evaluation
"""
from sklearn.metrics import r2_score

test_x_poly = poly.fit_transform(test_x)
test_y_ = clf.predict(test_x_poly)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )


#for example, if we want to use a polynomial regression with the dataset with the degree equals to 3.

ploy3 = PolynomialFeatures(degree=3)
x_train3 = ploy3.fit_transform(train_x)
clf3 = linear_model.LinearRegression()
y_train3 = clf3.fit(x_train3,train_y)
print ('Coefficients: ', clf3.coef_)
print ('Intercept: ', clf3.intercept_)
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf3.intercept_[0]+clf3.coef_[0][1]*XX+clf3.coef_[0][2]*np.power(XX, 2)+clf3.coef_[0][3]*np.power(XX,3)
plt.plot(XX,yy,'-r')
x_test3 = ploy3.fit_transform(test_x)
y_test3 = clf3.predict(x_test3)
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_test3 - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_test3 - test_y) ** 2))
print('R-squae: %.2f' % r2_score(y_test3, test_y))