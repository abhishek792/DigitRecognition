__author__ = 'abchauhan'
import scipy
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

columnNames = []
for i in range(783):
    columnNames.append('pixel' + str(i))

train = pd.read_csv('C:\\Users\\abchauhan\\Downloads\\train.csv')
trainData = train.loc[0:24998, columnNames]  # Training set
target = train['label']
print('First three true labels are: ', target[25000], target[25001], target[25002])
targetData = target[0:24999]  # I have one doubt here can discuss on phone.
testData = train.loc[25000:41999, columnNames]  # Cross-validation set

rf = RandomForestClassifier(n_estimators=150, min_samples_split=2, n_jobs=-1)
print('Fitting the data')
rf.fit(trainData, targetData)
print('Data fitted now predicting')
predicted = rf.predict(testData)

j = 0
correctPrediction = 0
for i in range(25000, 42000):
    if target[i] == predicted[j]:
        correctPrediction += 1
    j += 1
print('First three predictions are: ', predicted[0], predicted[1], predicted[2])
print('Accuracy is ', (correctPrediction / 17000) * 100)