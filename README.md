# Handwritten_Digit_Recognition_ANN
Using MNIST dataset of images to create an Artificial Neural Network to identify Handwritten Digits

Usage: 

      mabouc4699_Project_Two

On execution, the MNIST training and testing data sets and labels are loaded into variables tr_set, tr_labels, ts_set, and ts_labels respectively. After, 
the program will begin to save essential data values to variables in preparation for the creation of the ANN. The ANN is created using 60 default neurons
and saves the respective trained ANN to a cell matrix array. The resulting NN is tested against the ts_set, and the plotregression of the training data and 
the 20% or 1/5 reserved data is displayed for user analysis. Lastly, 80 images are displayed on a window of subplots with their "learned" label value.

Note: On execution, the testing_accuracy.m file will calculate for a given data set the accuracy of the NN predicted label(s) using the confusion matrix 
((TP + TN) / (TP + TN + FP + FN) = Accuracy of the data set)). Purpose of this file being further systematic testing of the NN accuracy.
