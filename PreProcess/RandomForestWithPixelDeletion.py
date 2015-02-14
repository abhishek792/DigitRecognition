__author__ = 'abchauhan + anavil :P'
# import scipy
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


REDUCTION = 3
NO_OF_DATA = 42000
NO_OF_TRAIN = 500
NO_OF_TEST = 250

columnNames = []
for i in range(783):
    columnNames.append('pixel' + str(i))

train = pd.read_csv('/home/anavil/Programming/DigitRecongnizer/train.csv')

finalLoda = []

trainData = train.loc[0:24998, columnNames]

for j in range(0, NO_OF_TRAIN):

    newImage= []
    for column in columnNames:
        newImage.append(pd.Series(trainData[column]).iloc[j])

    newImage = newImage[28*REDUCTION:-28*REDUCTION]

    newArray = []
    for columncutter in range(0,len(newImage), 28 ):
        newArray.append(newImage[REDUCTION+columncutter:(28-REDUCTION)+columncutter])


    lodu = [item for sublist in newArray for item in sublist]
    # print("done again")
    finalLoda.append(lodu)

trainData = finalLoda



# trainData = train.loc[0:24998, columnNames]  # Training set
target = train['label']
print('First three true labels are: ', target[25000], target[25001], target[25002])
targetData = target[0:NO_OF_TRAIN]  # I have one doubt here can discuss on phone.

testData = train.loc[25000:41999, columnNames]  # Cross-validation set

finalLoda = []

for j in range(0, NO_OF_TEST):

    newImage= []
    for column in columnNames:
        newImage.append(pd.Series(testData[column]).iloc[j])

    newImage = newImage[28*REDUCTION:-28*REDUCTION]

    newArray = []
    for columncutter in range(0,len(newImage), 28 ):
        newArray.append(newImage[REDUCTION+columncutter:(28-REDUCTION)+columncutter])


    lodu = [item for sublist in newArray for item in sublist]
    # print("done again")
    finalLoda.append(lodu)

testData = finalLoda

rf = RandomForestClassifier(n_estimators=150, min_samples_split=2, n_jobs=-1)
print('Fitting the data')
rf.fit(trainData, targetData)
print('Data fitted now predicting')
predicted = rf.predict(testData)

j = 0
correctPrediction = 0
k=0
for i in range(25000, 25000+NO_OF_TEST):
    k+=1
    print(k)
    if target[i] == predicted[j]:
        correctPrediction += 1
    j += 1
    
print(correctPrediction)
print('First three predictions are: ', predicted[0], predicted[1], predicted[2])

correctPrediction/= 17000
correctPrediction*=100
print(correctPrediction)
