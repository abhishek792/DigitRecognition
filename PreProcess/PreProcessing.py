__author__ = 'abchauhan'
#Some trivial preprocessing. Finding out columns which are always zero.
import pandas as pd

columnNames = []
for i in range(783):
    columnNames.append('pixel' + str(i))

train = pd.read_csv('C:\\Users\\abchauhan\\Downloads\\train.csv')
trainData = train.loc[:, columnNames]
nonZeroColumnValues = 0
colWithAtLeastOneNonZeroValue = 0
columnAsList = []
for column in columnNames:
    columnAsList = list(trainData[column])
    for colValues in columnAsList:
        if colValues > 0:
            nonZeroColumnValues += 1
            break
    if nonZeroColumnValues > 0:
        colWithAtLeastOneNonZeroValue += 1
    nonZeroColumnValues = 0
print(colWithAtLeastOneNonZeroValue)



