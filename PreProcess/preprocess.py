
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from PIL import Image
import numpy
#import csv
import pandas as pd

train = pd.read_csv('/home/anavil/Programming/DigitRecongnizer/train.csv')

columnNames = []
for i in range(784):
    columnNames.append('pixel' + str(i))

trainData = train.loc[:, columnNames]

i =0
newImage= []

for column in columnNames:
    newImage.append(pd.Series(trainData[column]).iloc[5])

finalImage=[]
for i in range(0,len(newImage),28):
    finalImage.append(newImage[i:i+27])

image_trial = numpy.array(finalImage)
im = Image.fromarray(image_trial.astype('uint8'))
plt.imshow(im)
#plt.axis('off')
plt.show()
#print(image_trial)
#plt.imshow(image_trial)

#image_file = cbook.get_sample_data("/home/anavil/Programming/DigitRecongnizer/pycharm.png")
#image = plt.imread(image_file)



