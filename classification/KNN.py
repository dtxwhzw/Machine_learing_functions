'''import the library'''
from sklearn.neighbors import KNeighborsClassifier
'''Training, let's start with k = 4 this time'''
k = 4
#Train Model and Predict
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh
'''And we use the model to predict the test set:'''
yhat = neigh.predict(X_test)
yhat[0:5]
'''
Accuracy evaluation
In multilabel classification, accuracy classification score is a function that computes subset accuracy. 
This function is equal to the jaccard_similarity_score function. Essentially, it calculates how closely 
the actual labels and predicted labels are matched in the test set.
'''
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
'''
What about other K?¶
K in KNN, is the number of nearest neighbors to examine. It is supposed to be specified by the User. 
So, how can we choose right value for K? The general solution is to reserve a part of your data for testing 
the accuracy of the model. Then chose k =1, use the training part for modeling, and calculate the accuracy of 
prediction using all samples in your test set. Repeat this process, increasing the k, and see which k is the best for your model.

We can calculate the accuracy of KNN for different Ks.
'''
Ks = 10
mean_acc = np.zeros((Ks - 1))
std_acc = np.zeros((Ks - 1))
ConfustionMx = [];
for n in range(1, Ks) :
    # Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n - 1] = metrics.accuracy_score(y_test, yhat)

    std_acc[n - 1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])

print(mean_acc)
'''Plot model accuracy for Different number of Neighbors'''
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
'''
tight_layout() used to adjust the parameter of the subplot, it will check the label of the plot and ensure them to
show entirely.
plt.fill_between(x, 0, y, facecolor='green', alpha=0.3) used to fill some color into the plot.
the x means to range in x_axis, the second parameter is the lower limit, at this time, it means form the x_axis,
the third parameter is the upper limit. and the facecolor is the color, and alpha is the transparency.
'''
plt.show()