__author__ = 'anavil'

import time
start_time =  time.time()
from sklearn import svm
import pandas as pd

#Constants
REDUCTION = 3
NO_OF_DATA = 42000
CURRENT_SET  = 41999
NO_OF_TRAIN = 24999
NO_OF_TEST = 16999 # CURRENT_SET-NO_OF_TRAIN


columnNames = []
for i in range(783):
    columnNames.append('pixel' + str(i))

train = pd.read_csv('C:\\Users\\abchauhan\\Downloads\\train.csv')

trainData = train.loc[0:NO_OF_TRAIN-1, columnNames]  # Training set
target = train['label']
print('First three true labels are: ', target[NO_OF_TRAIN+1], target[NO_OF_TRAIN+2], target[NO_OF_TRAIN+3])
targetData = target[0:NO_OF_TRAIN]
testData = train.loc[NO_OF_TRAIN+1:CURRENT_SET, columnNames]  # Cross-validation set

#Create a classifier: a support vector classifier
classifier = svm.SVC(C=10000, gamma=0.0000001)

print('Fitting the data')
#Learning on TrainData
classifier.fit(trainData, targetData)

print('Data fitted now predicting')
#Now prediction
predicted = classifier.predict(testData)

j = 0
correctPrediction = 0
for i in range(NO_OF_TRAIN+1, NO_OF_TRAIN + 1 + NO_OF_TEST):
    if target[i] == predicted[j]:
        correctPrediction += 1
    j += 1

print('First three predictions are: ', predicted[0], predicted[1], predicted[2])

print("Accuracy = ",  float(correctPrediction)/NO_OF_TEST*100)

print("Execution time was ", time.time()-start_time)
