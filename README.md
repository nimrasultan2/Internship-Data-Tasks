# Feature Engineering Challenge 

## Project Objective
The goal of this task is to show how we can make a Machine Learning model better by creating new data columns (Engineering) and picking only the most useful ones (Selection).

What the Code Does
The script goes through 5 clear sections to improve the model:

### Section 1: Load Data

We use the California Housing dataset (built into Python).
It starts with 8 basic features like house age and average rooms.

### Section 2: Baseline Model

We train a Random Forest on the original 8 features.
We save this "Baseline Score" so we have a starting point to compare against.

### Section 3: Create New Features (Engineering)
We create 3 new columns to help the model see better patterns:
Interaction: Rooms_per_Household (Total rooms / Total people).
Polynomial: Income_Squared (Squaring the median income).
Binning: Age_Group (Dividing house ages into 4 groups: 1, 2, 3, 4).

### Section 4: Feature Selection (RFE)

We now have 11 features, but some might be "noise."
We use RFE (Recursive Feature Elimination) to automatically fight the features against each other.
The system picks only the Top 5 most important columns.

### Section 5: Final Comparison

We train the model again using only those 5 best features.


### Libraries Used
| Library | Purpose |
| :--- | :--- |
| **pandas** | Data handling and management |
| **numpy** | Numerical operations and dataset generation |
| **scikit-learn** | ML models, TF-IDF, and Feature Selection (RFE) |
| **pickle** | Saving and loading the trained model |
| **streamlit** | Creating the Graphical User Interface (GUI) |

### Result: The code prints a table showing that even with fewer features, the model stays strong because the new engineered data is higher quality.
