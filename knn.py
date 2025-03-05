#-------------------------------------------------------------------------
# AUTHOR: Keon Der
# FILENAME: knn.py
# SPECIFICATION: Implementing 1-Nearest Neighbor (1NN) classifier for binary classification
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1.5 hours
#-----------------------------------------------------------*/
# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH
# AS numpy OR pandas. You have to work here only with standard vectors and arrays
# Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv


db = []

# Reading the data in a csv file
with open('email_classification.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            db.append(row)

    # Change ham to 0 and spam to 1
class_mapping = {'ham': 0, 'spam': 1}

# Loop data to allow each instance to be the test set
error_count = 0
for i, test_instance in enumerate(db):
    # Get features and labels for training data (excluding current test instance)
    X = []
    Y = []
    for j, row in enumerate(db):
        if j != i:  # Skip the test instance
            X.append([float(value) for value in row[:-1]])  #Features
            # Map ham and spam to 0 and 1
            Y.append(class_mapping[row[-1]])

    # Turn the original training classes to numbers and add them to the vector Y
    test_sample = [float(value) for value in test_instance[:-1]]  # Test instance features
    true_label = class_mapping[test_instance[-1]]  # True class label

    # Fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    # Use the test sample in this iteration to make the class prediction
    class_predicted = clf.predict([test_sample])[0]

    # Compare the prediction with the true label to calculate the error rate
    if class_predicted != true_label:
        error_count += 1

error_rate = error_count / len(db)
print(f"Error Rate: {error_rate * 100:.2f}%")