import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from pickle import dump

wine_df = pd.read_csv('winequality-red.csv')

X = wine_df.drop("quality", axis=1)
y = wine_df["quality"].apply(lambda y_value:1 if y_value>=6 else 0)

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the scaler with only training data
scaler = StandardScaler().fit(X_train.values)

# Scale both the training and test data.
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model with the scaled data
model = LogisticRegression(random_state=42).fit(X_train_scaled, y_train)

# Predict the test data, compare with the real values and print the classification report
y_pred = model.predict(X_test_scaled)

print(classification_report(y_test, y_pred))

dump(model, open('model.pkl', 'wb'))
dump(scaler, open('scaler.pkl', 'wb'))