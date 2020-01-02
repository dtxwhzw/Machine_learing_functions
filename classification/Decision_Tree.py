"""import the library"""
from sklearn.tree import DecisionTreeClassifier

"""
Setting up the Decision Tree
We will be using train/test split on our decision tree. Let's import train_test_split from sklearn.cross_validation.
"""
from sklearn.model_selection import train_test_split

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
'''
Now train_test_split will return 4 different parameters. We will name them:
X_trainset, X_testset, y_trainset, y_testset

The train_test_split will need the parameters:
X, y, test_size=0.3, and random_state=3.
'''
"""
let's print the shape of these dataset to ensure that the dimensions match
the output would be like (60,5) (60,) etc.
"""
print(X_trainset.shape)
print(y_trainset.shape)
print(X_testset)
print(y_testset)
"""
Modeling
We will first create an instance of the DecisionTreeClassifier.
Inside of the classifier, specify criterion="entropy" so we can see the information gain of each node.
"""
Tree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
Tree  # it shows the default parameters
'''
Next, we will fit the data with the training feature matrix X_trainset and training response vector y_trainset
'''
Tree.fit(X_trainset, y_trainset)
'''
Prediction
Let's make some predictions on the testing dataset and store it into a variable called predTree.
'''
predTree = drugTree.predict(X_testset)
'''
Evaluation
Next, let's import metrics from sklearn and check the accuracy of our model.
'''
from sklearn import metrics
import matplotlib.pyplot as plt

print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))
'''
Accuracy classification score computes subset accuracy: the set of labels predicted for a sample must exactly 
match the corresponding set of labels in y_true.

In multilabel classification, the function returns the subset accuracy. If the entire set of predicted labels 
for a sample strictly match with the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0.
'''

"""
Visualization
"""
# Notice: You might need to uncomment and install the pydotplus and graphviz libraries if you have not installed these before
# !conda install -c conda-forge pydotplus -y
# !conda install -c conda-forge python-graphviz -y


from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

# %matplotlib inline (you dont need it unless you're using jupyter notebook)


dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0 :5]
targetNames = my_data["Drug"].unique().tolist()
out = tree.export_graphviz(Tree, feature_names=featureNames, out_file=dot_data, class_names=np.unique(y_trainset),
                           filled=True, special_characters=True, rotate=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img, interpolation='nearest')
