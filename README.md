Task: Credit Risk Prediction

The goal of this task was to build a classification model that predicts whether a loan applicant will default on their loan.
I addressed missing values by using the median for numerical data (like Loan Amount) and the mode (most frequent value) for categorical data.
I created visualizations to understand the data whch i learnt in Task02: 
         Histograms: To see the common ranges of requested loan amounts.
         Count Plots: To compare how education levels impact loan approval.
         Box Plots: To identify if higher applicant income correlates with lower default rates.
I implemented a Logistic Regression model, which is ideal for predicting binary outcomes (1 for approved, 0 for defaulted).
I measured success using an Accuracy Score and a Confusion Matrix to see exactly how many True Positives and False Negatives the model produced.
