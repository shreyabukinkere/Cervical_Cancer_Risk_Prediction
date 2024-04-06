## Cervical Cancer Risk Predictor: A Machine Learning Approach
This project aims to develop a machine learning model for predicting the risk of cervical cancer based on global, clinical data. Cervical cancer is a significant health concern globally, and early prediction of risk factors can facilitate timely interventions and preventive measures.
## Libraries used:
import pandas as pd : Data Manipulation ; 
import numpy as np : used for creating multidimentional arrays ;
import seaborn as sns : used for data intensity viaualisation, heatmaps ;
import matplotlib.pyplot as plt : used for graphs ;
import zipfile ;
!pip install plotly ;
!pip install jupyterthemes ;
import plotly.express as px ;
from jupyterthemes import jtplot ;
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False) ;
## Data Used and preprocessing 
The dataset used is from Kaggle.
In this project we preprocess the data by removing all the null values, filling in missing data, analysing the correlation matrix, standardising it and then continue to train and test the data.
## Algorithm used: XGBoost Algorithm
XGBoost stands for eXtreme Gradient Boosting, a powerful machine learning algorithm known for its speed and performance.
It belongs to the ensemble learning family, specifically boosting algorithms, which combine the predictions of several weak learners to improve overall accuracy.
XGBoost is highly efficient due to its optimized implementation and parallelization capabilities, making it suitable for large-scale datasets.
It supports both regression and classification tasks, making it versatile for various predictive modeling problems.
Key features include regularization techniques to prevent overfitting and handling missing data effectively.
## Results
From sklearn.metrics we import confusion_matrix, classification_report. The classification_report function generates a text report showing the main classification metrics (precision, recall, F1-score, and support) for each class and the confusion_matrix function computes a confusion matrix to evaluate the accuracy of a classification model. It takes the true labels (y_true) and predicted labels (y_pred) as input.


