# README: Subscription Prediction Model

This README explains every single line of code in the provided script in a beginner-friendly way, including why we use certain steps and what each component does. This guide is perfect for understanding the logic behind the code.

---
## Introduction
The goal of this script is to predict whether a customer will renew their subscription or not, based on their signup date and cancellation status. This is achieved using a **Logistic Regression model**. Logistic Regression is a simple yet powerful machine learning algorithm used for binary classification problems (like predicting 1 or 0).

---
## Step-by-Step Explanation

### 1. Importing Necessary Libraries
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
```
- **pandas (pd)**: Helps in handling and manipulating tabular data (like spreadsheets).
- **scikit-learn (sklearn)**: Provides machine learning tools for tasks like splitting data, scaling features, training models, and evaluating results.
- **numpy (np)**: Helps with numerical computations and working with arrays.

### 2. Loading the Dataset
```python
data = pd.read_csv('Subscription Prediction\Dataset\customer_product.csv')
```
- Loads the dataset into a pandas DataFrame.
- **Why?** This is the data we need to train the model and make predictions. Data is the foundation of any machine learning project.

### 3. Adding a New Column for Subscription Status
```python
data['cancel_date_time'] = pd.to_datetime(data['cancel_date_time'], errors='coerce')
data['renewal_status'] = np.where(data['cancel_date_time'].isna(), 1, 0)
```
- Converts the `cancel_date_time` column into a datetime format to handle date-related operations.
- Creates a new column, `renewal_status`: 
  - **1** if the subscription is active (no cancellation date).
  - **0** if the subscription is canceled.
- **Why?** We need to classify customers into two categories: those who renew (1) and those who don’t (0).

### 4. Extracting Signup Month
```python
data['signup_date_time'] = pd.to_datetime(data['signup_date_time'])
data['signup_month'] = data['signup_date_time'].dt.month
```
- Converts the `signup_date_time` column to a datetime format.
- Extracts the **month** from the signup date.
- **Why?** The month of signup could influence the likelihood of renewal. For instance, customers who sign up in festive months might behave differently.

### 5. Defining Features and Target
```python
X = data[['signup_month']]
y = data['renewal_status']
```
- **X** (features): Data used to make predictions. Here, it’s the signup month.
- **y** (target): The output we’re predicting—whether the customer renews their subscription.
- **Why?** Machine learning models need input (X) and output (y) to learn patterns.

### 6. Splitting Data into Training and Testing Sets
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```
- Splits the dataset into training (80%) and testing (20%) sets.
- **stratify=y** ensures the same proportion of classes (1 and 0) in both sets.
- **Why?** To train the model on one part of the data and evaluate it on unseen data.

### 7. Scaling the Features
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
- Standardizes the features by removing the mean and scaling to unit variance.
- **Why?** Machine learning models perform better when input features are on the same scale.

### 8. Training the Logistic Regression Model
```python
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
```
- Creates a Logistic Regression model.
- Fits (trains) the model on the scaled training data.
- **Why?** Logistic Regression is great for binary classification problems like this one.

### 9. Evaluating the Model
```python
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
```
- Predicts the renewal status for the test set.
- Calculates the model’s accuracy.
- **Why?** To check how well the model performs on unseen data.

### 10. Taking User Input
```python
user_signup = input("Enter the user signup date in YYYY-MM-DD format: ")
user_cancel = input("Enter the user cancellation date (leave blank if not canceled): ")
```
- Takes the user’s signup and cancellation dates as input.
- **Why?** To predict renewal status for new users dynamically.

### 11. Processing User Input
```python
try:
    user_signup = pd.to_datetime(user_signup)
    user_month = user_signup.month
except Exception as e:
    print("Invalid signup date format. Please use YYYY-MM-DD.")
    exit()

if user_cancel.strip() == "":
    print("User has not canceled the subscription.")
else:
    try:
        user_cancel = pd.to_datetime(user_cancel)
        print("User cancellation date recorded.")
    except Exception as e:
        print("Invalid cancellation date format. Please use YYYY-MM-DD.")
        exit()
```
- Converts the user’s signup date to datetime and extracts the month.
- Checks if the cancellation date is blank or valid.
- **Why?** Ensures valid input for making predictions.

### 12. Making Predictions for User Input
```python
user_input_scaled = scaler.transform([[user_month]])
user_prediction = model.predict(user_input_scaled)[0]

if user_prediction == 1:
    print("User is predicted to renew their subscription!")
else:
    print("User is predicted not to renew their subscription.")
```
- Scales the user’s input month.
- Predicts the renewal status (1 or 0).
- Prints the result.
- **Why?** Provides a prediction based on the user’s details.

---
## Tips and Tricks
1. **Always Validate Input:** Use `try-except` to handle invalid user input gracefully.
2. **Feature Scaling:** Ensure all numerical features are scaled for optimal model performance.
3. **Test Accuracy:** A good model balances accuracy and interpretability.
4. **Stratified Splitting:** Maintain class balance in train-test splits for unbiased evaluation.

---
## Conclusion
This script demonstrates a complete pipeline—from loading data to making predictions—using Logistic Regression. It includes robust practices like feature scaling, stratified splitting, and handling user input for real-world applications.

---
