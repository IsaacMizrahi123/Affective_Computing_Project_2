# Affective_Computing_Project_2
Python script that identifies (classify) pain from images, using Machine Learning. This project uses Tensorflow Keras and sklearn metrics.

To run the script: python Project2.py W H dataDirectory

Where W is the width of an image (integer); H is the height of an image (integer); and dataDirectory is the directory where data is located. This is an absolute directory. Don’t make it relative to the script.

This script use deep learning to classify the images as either pain or no pain.

The Data directory has to follow a hierarchy as follows:
Project2Data
* Testing
  - Pain
  - Image data
  - No_pain
  - Image data
* Training
  - Pain
  - Image data
  - No_pain
  - Image data
* Validation
  - Pain
  - Image data
  - No_pain
  - Image data

The output of the script prints the confusion matrix, classification accuracy, precission, recall, and binary F1 score for the test set.

A PDF file with the results of the project is included.
