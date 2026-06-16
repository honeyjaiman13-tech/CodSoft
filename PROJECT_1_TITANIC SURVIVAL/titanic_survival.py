import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================================
# 1. LOAD THE DATASET
# ==========================================
print("--- Loading Titanic Dataset ---")
df = pd.read_csv('Titanic-Dataset.csv')

# Print the first 5 rows to the console window
print("\nFirst 5 rows of the dataset:")
print(df.head())

# ==========================================
# 2. EXPLORATORY DATA ANALYSIS & VISUALIZATION
# ==========================================
print("\n--- Analyzing & Visualizing Data ---")
# Check for missing values across columns
print("\nMissing values per column:")
print(df.isnull().sum())

# Generate a survival count chart grouped by Gender
print("\nSaving gender-based survival analysis chart as 'survival_by_gender.png'")
plt.figure(figsize=(7, 5))
sns.countplot(x='Survived', hue='Sex', data=df, palette='Set2')
plt.title('Survival Count Distribution by Gender')
plt.xlabel('Survival Status (0 = No, 1 = Yes)')
plt.ylabel('Passenger Count')
plt.savefig('survival_by_gender.png', bbox_inches='tight')
plt.close()

# ==========================================
# 3. DATA PREPROCESSING & CLEANING
# ==========================================
print("\n--- Cleaning & Preprocessing Data ---")

# Fill missing values for Age with the median age of passengers
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill missing values for Embarked with the most common port ('S')
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop unique identifier columns that do not contribute to machine learning generalizations
X = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'])
y = df['Survived']

# Convert categorical columns ('Sex' and 'Embarked') into numerical dummy variables
# This replaces text labels with numeric true/false indicator columns
X = pd.get_dummies(X, columns=['Sex', 'Embarked'], drop_first=True)

# Split the dataset: 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]} rows")
print(f"Testing set size: {X_test.shape[0]} rows")

# ==========================================
# 4. MODEL TRAINING
# ==========================================
print("\n--- Training Random Forest Classifier ---")
# Initialize the model with optimized hyper-parameters to prevent overfitting
model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)

# Train the model
model.fit(X_train, y_train)

# ==========================================
# 5. MODEL EVALUATION
# ==========================================
print("\n--- Evaluation Metrics ---")
# Generate predictions on the testing dataset
y_pred = model.predict(X_test)

# Calculate and print overall classification accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Generate detailed metrics (Precision, Recall, F1-Score)
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Generate and save the Confusion Matrix heatmap
print("Generating Confusion Matrix... Saving as 'titanic_confusion_matrix.png'")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Perished', 'Survived'], yticklabels=['Perished', 'Survived'])
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Titanic Survival Prediction Confusion Matrix')
plt.savefig('titanic_confusion_matrix.png', bbox_inches='tight')
plt.close()

print("\n--- Task 1 Completed Successfully! ---")
