from sklearn import svm
#Deleting a fixed amount of pixels from an image (chopping image from all the corners) and
#making all non zero values as 255.
__author__ = 'abchauhan + anavil'
# import scipy
import time
start_time =  time.time()
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


REDUCTION = 3
NO_OF_DATA = 42000
NO_OF_TRAIN = 24999
NO_OF_TEST = 16999

columnNames = []
for i in range(783):
    columnNames.append('pixel' + str(i))

train = pd.read_csv('C:\\Users\\abchauhan\\Downloads\\train.csv')

finalProcessedData = []

trainData = train.loc[0:24998, columnNames]

#Creation of trainData list out of trainData Data Frame by row wise extraction of values of every single column
#and converting non zero values to 255
trainDataCalTime = time.time()

for j in range(0, NO_OF_TRAIN):

    newImage = []
    for column in columnNames:
        newImage.append(0 if 0 == pd.Series(trainData[column]).iloc[j] else 255)
    newImage = newImage[28 * REDUCTION:-28 * REDUCTION]
    newArray = []
    for columnCutter in range(0, len(newImage), 28):
        newArray.append(newImage[REDUCTION + columnCutter:(28 - REDUCTION) + columnCutter])
    singleItem = [item for subList in newArray for item in subList]
    finalProcessedData.append(singleItem)

trainData = finalProcessedData
print("Time to calculate train data ",time.time()- trainDataCalTime)

target = train['label']
print('First three true labels are: ', target[25000], target[25001], target[25002])
targetData = target[0:NO_OF_TRAIN]

testData = train.loc[25000:41999, columnNames]  # Cross-validation set

finalProcessedData = []

#Creation of testData list out of testData Data Frame by row wise extraction of values of every single column
#and converting non zero values to 255
testDataCalcTime = time.time()
for j in range(0, NO_OF_TEST):
    newImage = []
    for column in columnNames:
        newImage.append(0 if pd.Series(testData[column]).iloc[j]==0 else 255) #value extraction and conversion to binary

    newImage = newImage[28 * REDUCTION:-28 * REDUCTION]

    newArray = []
    for columnCutter in range(0, len(newImage), 28):
        newArray.append(newImage[REDUCTION + columnCutter:(28 - REDUCTION) + columnCutter])

    singleItem = [item for sublist in newArray for item in sublist]
    # print("done again")
    finalProcessedData.append(singleItem)

testData = finalProcessedData
print("time to calculate test data", time.time()- testDataCalcTime)

timeForRealWork = time.time()
rf = RandomForestClassifier(n_estimators=150, min_samples_split=2, n_jobs=-1) #For random forest
#classifier = svm.SVC(C=10000, gamma=0.0000001) #For SVM
print('Fitting the data')
rf.fit(trainData, targetData) #For random forest
#classifier.fit(trainData, targetData) #For SVM
print('Data fitted now predicting')
predicted = rf.predict(testData) #For random forest
#predicted = classifier.predict(testData) #For SVM

print("Time taken for real work ", time.time()- timeForRealWork)
j = 0
correctPrediction = 0
for i in range(25000, 25000 + NO_OF_TEST):
    if target[i] == predicted[j]:
        correctPrediction += 1
    j += 1

print(correctPrediction)
print('First three predictions are: ', predicted[0], predicted[1], predicted[2])

print(float(correctPrediction)/NO_OF_TEST*100)

print("Execution time was ", time.time()-start_time)
