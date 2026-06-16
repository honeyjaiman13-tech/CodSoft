import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. LOAD THE DATASET
# ==========================================
print("--- Loading Advertising Dataset ---")
# Read the CSV file (using the exact name uploaded)
df = pd.read_csv('advertising.csv.csv')

# Print the first 5 rows to the console window
print("\nFirst 5 rows of the dataset:")
print(df.head())

# ==========================================
# 2. EXPLORATORY DATA ANALYSIS & VISUALIZATION
# ==========================================
print("\n--- Analyzing & Visualizing Data ---")
# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Display basic statistics of the budget and sales
print("\nStatistical Summary:")
print(df.describe())

# Generate a correlation heatmap to show which platform impacts sales the most
print("\nGenerating correlation heatmap... Saving as 'advertising_correlations.png'")
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Analysis of Advertising Channels')
plt.savefig('advertising_correlations.png', bbox_inches='tight')
plt.close()

# ==========================================
# 3. DATA PREPROCESSING
# ==========================================
print("\n--- Preprocessing Data ---")
# Features (independent variables): budgets for TV, Radio, and Newspaper
X = df[['TV', 'Radio', 'Newspaper']]
# Target variable (dependent variable): Sales volume
y = df['Sales']

# Split the dataset: 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# ==========================================
# 4. MODEL TRAINING
# ==========================================
print("\n--- Training Linear Regression Model ---")
# Initialize the Linear Regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Print out the calculated intercept and coefficients
print(f"Model Intercept (Base Sales): {model.intercept_:.4f}")
print("Feature Coefficients (Impact per unit spent):")
for col, coef in zip(X.columns, model.coef_):
    print(f"  {col}: {coef:.4f}")

# ==========================================
# 5. MODEL EVALUATION
# ==========================================
print("\n--- Evaluation Metrics ---")
# Predict sales using the testing set
y_pred = model.predict(X_test)

# Calculate Evaluation Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R2 Score): {r2 * 100:.2f}%")

# Generate and save a scatter plot of Actual vs Predicted sales
print("\nGenerating Actual vs Predicted plot... Saving as 'actual_vs_predicted.png'")
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, edgecolors='black')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, linestyle='--')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs. Predicted Sales Volume')
plt.savefig('actual_vs_predicted.png', bbox_inches='tight')
plt.close()

print("\n--- Task 4 Completed Successfully! ---")
