#-------------------------------------------------------------------------
# AUTHOR: Keon der
# FILENAME: decision_tree.py
# SPECIFICATION: creates decision tree for contact_lens.csv
# FOR: CS 4210- Assignment #1
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)

#transform the original categorical training features into numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
for row in db:
    # Transform features into numeric values
    age = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3}.get(row[0], 0)

    spectacle = {'Myope': 1, 'Hypermetrope': 2}.get(row[1], 0)

    astigmatism = {'Yes': 1, 'No': 2}.get(row[2], 0)

    tear = {'Reduced': 1, 'Normal': 2}.get(row[3], 0)

    X.append([age, spectacle, astigmatism, tear])
# X =

#transform the original categorical training classes into numbers and add to the vector Y. For instance Yes = 1, No = 2
for row in db:
    class_value = {'Yes': 1, 'No': 2}.get(row[4], 0)
    Y.append(class_value)
# Y =

#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()