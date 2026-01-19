import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
df = pd.read_csv(r'C:/Users/nimra/Downloads/train_u6lujuX_CVtuZ9i (1).csv')

#Missing Data Handling
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

#Filling missing values
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])

#Visualization (EDA)
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(df['LoanAmount'], kde=True, color='green')
plt.title('Loan Amount Distribution')

plt.subplot(1, 3, 2)
sns.countplot(x='Education', hue='Loan_Status', data=df)
plt.title('Education vs Loan Status')

plt.subplot(1, 3, 3)
sns.boxplot(y=df['ApplicantIncome'], x=df['Loan_Status'])
plt.title('Income vs Loan Status')
plt.tight_layout()
plt.show()

#Converting strings to numbers to make computer understand 
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
#To ttrain model
X = df[['ApplicantIncome', 'LoanAmount', 'Education', 'Credit_History']]
y = df['Loan_Status']
#split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#logisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nAccuracy Score:", accuracy_score(y_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nDetailed Report:\n", classification_report(y_test, predictions))