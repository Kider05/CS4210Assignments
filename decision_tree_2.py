#-------------------------------------------------------------------------
# AUTHOR: Keon Der
# FILENAME: decision_tree_2.py
# SPECIFICATION: Train and evaluate decision trees on different datasets
# FOR: CS 4210 - Assignment #2
# TIME SPENT: 1.5 hours
#-----------------------------------------------------------*/
# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY SUCH AS numpy OR pandas.

# Importing required libraries
from sklearn import tree
import csv

# Training datasets
dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

# Mapping categorical values to numerical values
def encode_features(row):
    mapping = {
        'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3,
        'Myope': 1, 'Hypermetrope': 2,
        'No': 1, 'Yes': 2,
        'Reduced': 1, 'Normal': 2
    }
    return [mapping[val] for val in row[:-1]], mapping[row[-1]]

# Process each dataset
for ds in dataSets:
    dbTraining = []
    X = []
    Y = []

    # Reading the training data from CSV file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0:  # Skipping the header
                dbTraining.append(row)  # Store raw data
                x, y = encode_features(row)  # Convert categorical to numeric
                X.append(x)
                Y.append(y)

    # Loop to train and test the model 10 times
    total_accuracy = 0
    for i in range(10):
        # Train decision tree with max_depth=5
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
        clf = clf.fit(X, Y)

        # Read test data
        dbTest = []
        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                dbTest.append(row)

        # Evaluate on test data
        correct_predictions = 0
        for data in dbTest:
            x_test, y_true = encode_features(data)
            class_predicted = clf.predict([x_test])[0]  # Predict class
            if class_predicted == y_true:
                correct_predictions += 1

        # Compute accuracy
        accuracy = correct_predictions / len(dbTest)
        total_accuracy += accuracy

    # Compute and print final average accuracy
    final_accuracy = total_accuracy / 10
    print(f"Final accuracy when training on {ds}: {final_accuracy:.4f}")
