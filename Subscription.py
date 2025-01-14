import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Load the dataset
data = pd.read_csv('Subscription Prediction\Dataset\customer_product.csv')

# Step 1: Create a 'renewal_status' column based on 'cancel_date_time'
# Active user = 1, If canceled = 0
data['cancel_date_time'] = pd.to_datetime(data['cancel_date_time'])  # Convert to datetime
data['renewal_status'] = np.where(data['cancel_date_time'].isna(), 1, 0)

# Step 2: Feature Engineering
# Convert 'signup_date_time' to 'signup_month' (month of signup)
data['signup_date_time'] = pd.to_datetime(data['signup_date_time'])
data['signup_month'] = data['signup_date_time'].dt.month

# Step 3: Define features (X) and target (y)
X = data[['signup_month']]  # Only 'signup_month' as a feature
y = data['renewal_status']  # 'renewal_status' as the target variable

# Step 4: Train-test split with stratification to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 5: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.4f}')

# Step 8: Predict for a new user
user_signup = input("Enter the user signup date in YYYY-MM-DD format: ")
user_cancel = input("Enter the user cancellation date (leave blank if not canceled): ")

try:
    user_signup = pd.to_datetime(user_signup)  # Convert user signup date to datetime
    user_month = user_signup.month  # Extract the month for prediction
except Exception as e:
    print("Invalid signup date format. Please use YYYY-MM-DD.")
    exit()

# Handle cancellation date input
if user_cancel.strip() == "":
    print("User has not canceled the subscription.")
else:
    try:
        user_cancel = pd.to_datetime(user_cancel)  # Convert user cancellation date to datetime
        print("User cancellation date recorded.")
    except Exception as e:
        print("Invalid cancellation date format. Please use YYYY-MM-DD.")
        exit()

# Convert the user's month input into a DataFrame
user_input_df = pd.DataFrame({'signup_month': [user_month]})  # Match the training feature names

# Scale the user's input
user_input_scaled = scaler.transform(user_input_df)


# Predict using the model
user_prediction = model.predict(user_input_scaled)[0]  # Get the prediction (renewal status)

# Print the prediction result
if user_prediction == 1:
    print("User is predicted to renew their subscription!")
else:
    print("User is predicted not to renew their subscription!")