#-------------------------------------------------------------------------
# AUTHOR: Keon Der
# FILENAME: naive_bayes.py
# SPECIFICATION: Implementing naive bayes
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1.5 hours
#-----------------------------------------------------------*/
# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH
# AS numpy OR pandas. You have to work here only with standard vectors and arrays

from sklearn.naive_bayes import GaussianNB
import csv

# Reading the training data in a csv file
training_data = []
with open('weather_training.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        training_data.append(row)

# Transform the original training features to numbers and add them to the 4D array X.
outlook_dict = {'Sunny': 1, 'Overcast': 2, 'Rain': 3}
temp_dict = {'Hot': 1, 'Mild': 2, 'Cool': 3}
humidity_dict = {'High': 1, 'Normal': 2}
wind_dict = {'Weak': 1, 'Strong': 2}

X = [[outlook_dict[row[1]], temp_dict[row[2]], humidity_dict[row[3]], wind_dict[row[4]]] for row in training_data]

# Transform the original training classes to numbers and add them to the vector Y.
play_tennis_dict = {'Yes': 1, 'No': 2}
Y = [play_tennis_dict[row[5]] for row in training_data]

# Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

# Reading the test data in a csv file
test_data = []
with open('weather_test.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        test_data.append(row)

# Printing the header of the solution
print("Day, Outlook, Temperature, Humidity, Wind, PlayTennis Prediction, Confidence")

# Use test samples to make probabilistic predictions if confidence is >= 0.75
for row in test_data:
    test_instance = [[outlook_dict[row[1]], temp_dict[row[2]], humidity_dict[row[3]], wind_dict[row[4]]]]
    prediction = clf.predict(test_instance)[0]
    probabilities = clf.predict_proba(test_instance)[0]
    confidence = max(probabilities)
    if confidence >= 0.75:
        predicted_label = 'Yes' if prediction == 1 else 'No'
        print(f"{row[0]}, {row[1]}, {row[2]}, {row[3]}, {row[4]}, {predicted_label}, {confidence:.4f}")
