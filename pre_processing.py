import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
%matplotlib inline#if you'll run this code in jupyter notebook, then this line is needed, otherwise you can comment it.

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