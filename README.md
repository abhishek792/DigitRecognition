# DigitRecognition
A python implementation of handwritten digit recognition.
Problem details: https://www.kaggle.com/c/digit-recognizer
Project has two parts: Implementation and Pre-processing. As of now, implementation part is having pre-processing in-built and 
separate pre-processing is not done. One can still go through the Pre-processing folder.
Two algorithms are implemented: Support Vector Machines and Random Forest Classifier. As pre-processing, we have clipped some 
pixels from all the four corners and all non-zero values have been converted to 255. In the separate Pre-processing folder, we
identified one more strategy to remove all those pixels which have not been filled any time. However, that didn't yield any 
signigicant improvement. 

Results:
Benchmark Random Forest Classifier: 96.27%
Random Forest with Pre-processing: 96.48%
Benchmark SVM: 97.45%
