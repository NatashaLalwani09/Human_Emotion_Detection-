# CS613 Final Project

## Human emotion detection from speech

- We consider the problem of identifying human emotions from speech samples. Our aim is to experiment with various methods of classifying emotions and better understand the most distinctive features of emotions

### Dataset

Speech audio-only files (16bit, 48kHz .wav) from the RAVDESS dataset. It contains 1440 files that include the Speech emotions - calm, happy, sad, angry, fearful, surprise, and disgust expressions.


## Required Dependencies

- numpy
- librosa
- tensorflow
- keras
- sklearn
- os
- matplotlib

## Files

This project contain following files and folder:

### EDA_final.ipynb

This contains the Exploratory Data analysis for the speech audio data.

### Data_creation.ipynb

This file contains code for creating training and testing data by extracting features from audio signals

### KNN_n.ipynb, LogisticRegression.ipynb, SVM.ipynb, RandomForest.ipynb, Ensemble.ipynb

These files trains, validates and tests respective models on our speech emotion dataset

## Steps to run

- Load data

  - Run the Data_creation.ipynb file to extract features from the audio files and generating test, train, validation csv files
  - We also train ad test on scaled features using MinMaxScaler which also happens in Data_creation.ipynb file. csv files for scaled data set are created

- Train and test model

  - Run the model files ( KNN_n.ipynb,LogisticRegression.ipynb ) for training and testing the models against the dataset

## Conclusions -

- Feature scaling seems to improve performance than non-scaled features.
- Ensemble model has the highest accuracy than any of the other models tested.

## Future Scope -

- Train and test Deep learning model to test the audio data
- Need to consider various methods like augmentation and test its impact on model performance
