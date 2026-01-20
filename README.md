Task: Credit Risk Prediction

The goal of this task was to build a classification model that predicts whether a loan applicant will default on their loan.
1. I addressed missing values by using the median for numerical data (like Loan Amount) and the mode (most frequent value) for categorical data.
2. I created visualizations to understand the data whch i learnt in Task02: 
        a. Histograms: To see the common ranges of requested loan amounts.
        b. Count Plots: To compare how education levels impact loan approval.
        c. Box Plots: To identify if higher applicant income correlates with lower default rates.
3. I implemented a Logistic Regression model, which is ideal for predicting binary outcomes (1 for approved, 0 for defaulted).
4. I measured success using an Accuracy Score and a Confusion Matrix to see exactly how many True Positives and False Negatives the model produced.


The goal was to move from "looking at past loans" to "predicting future ones." By using Logistic Regression, I created a system that takes new data and outputs a 1 (Approved) or a 0 (Rejected).
The Confusion Matrix is the ultimate proof of fulfillment. It shows:
    a. True Positives: People predicted to be safe who actually were safe.
    b. True Negatives: People predicted to be risky who actually were risky.
