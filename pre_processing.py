import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
#%matplotlib inline if you'll run this code in jupyter notebook, then this line is needed, otherwise you can comment it.

'''
import the data or download the data here
'''


#read the data and take a look at the dataset.
df = pd.read_csv("file_name")
df.head()

#Lets first have a descriptive exploration on our data.
df.describe()

#draw the plot
plt.xlabel()#set the title for the xlabel
plt.ylabel()#set the title for the ylabel
plt.scatter(x_label,y_label,color)#draw the scatter plot
plt.show()#show the plot

'''
creating the test and train dataset
we split the dataset in this step, suppose we set x% of the entire data for training, and the rest for testing.
we create a mask to select random rows using np.random.rand() function.
'''
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

"""
if we are going to do linear regression, the pre_procession part ends here.
"""

df['custcat'].value#this will return each value of the given colunm
df['custcat'].value_counts()#this will return the number of each value in a column
df.hist(column='income', bins=50)
'''this will draw a histogram use the data in the specific column, and the bins equals to the number of blocks in the plot'''

'''
Normalize Data
Data Standardization give data zero mean and unit variance, it is good practice, 
especially for algorithms such as KNN which is based on distance of cases:
'''
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]
'''
Train Test Split
'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
'''
the column number of x is equal to y, and the random_state have the same effect as random.seed(), it is used to 
ensure the dataset is split into a same one. otherwise the output will be different everytime you run the code.
'''
'''this is the end of pre-processing of knn classification'''