# STEP 1: Import the libraries we need
# ------------------------------------
import pandas as pd  # For handling data in tables
import numpy as np   # For mathematical operations
import seaborn as sns  # Provides the Titanic dataset
from sklearn.model_selection import train_test_split  # To split data
from sklearn.linear_model import LogisticRegression  # The prediction model
from sklearn.preprocessing import StandardScaler  # To scale numbers
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer  # To fill missing values

# STEP 2: Load the Titanic dataset
# ---------------------------------
print("=" * 50)
print("STEP 2: Loading the Titanic Dataset")
print("=" * 50)

# Load the dataset (it comes built-in with seaborn)
df = sns.load_dataset('titanic')

# Show the first 5 rows to see what the data looks like
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Show information about the dataset (columns, missing values, data types)
print("\nDataset Information:")
print(df.info())

# STEP 3: Select the features we want to use
# -------------------------------------------
print("\n" + "=" * 50)
print("STEP 3: Selecting Features for Prediction")
print("=" * 50)

features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
target = 'survived'

# Create a new dataframe with only the columns we need
df_clean = df[features + [target]].copy()

print(f"\nWe selected {len(features)} features to predict survival")
print(f"Features: {features}")
print(f"Target: {target}")

# STEP 4: Handle missing values (Data Cleaning)
# ----------------------------------------------
print("\n" + "=" * 50)
print("STEP 4: Cleaning the Data (Handling Missing Values)")
print("=" * 50)

# Check for missing values before cleaning
print("\nMissing values before cleaning:")
print(df_clean.isnull().sum())

# 4.1: Fill missing age values with the median age
# Why median? Because it's not affected by extreme values (outliers)
median_age = df_clean['age'].median()
print(f"\nFilling missing ages with median age: {median_age:.1f}")
df_clean['age'] = df_clean['age'].fillna(median_age)

# 4.2: Drop rows where 'embarked' is missing (only a few rows)
# We drop them because 'embarked' is important for prediction
df_clean = df_clean.dropna(subset=['embarked'])
print(f"Dropped rows with missing embarkation information")

# Check if we fixed all missing values
print("\nMissing values after cleaning:")
print(df_clean.isnull().sum())

# STEP 5: Convert text to numbers (Encoding)
# -------------------------------------------
print("\n" + "=" * 50)
print("STEP 5: Converting Text to Numbers")
print("=" * 50)

# Machine learning models only understand numbers, not text
# We need to convert 'sex' and 'embarked' to numbers

# 5.1: Convert 'sex' to numbers (male=0, female=1)
print("\nConverting 'sex' (male/female) to numbers...")
df_clean['sex_encoded'] = df_clean['sex'].map({'male': 0, 'female': 1})
print(df_clean[['sex', 'sex_encoded']].head())

# 5.2: Convert 'embarked' to numbers using one-hot encoding
# This creates separate columns for each embarkation port
print("\nConverting 'embarked' to numbers (one-hot encoding)...")
df_clean = pd.get_dummies(df_clean, columns=['embarked'], prefix='port')
print("Added columns for each embarkation port:")
print(df_clean.filter(like='port').head())

# Now we can drop the original text columns
df_clean = df_clean.drop(['sex'], axis=1)

# Show all columns we have now
print(f"\nAll columns now: {list(df_clean.columns)}")

# STEP 6: Prepare features (X) and target (y)
# --------------------------------------------
print("\n" + "=" * 50)
print("STEP 6: Preparing Features and Target")
print("=" * 50)

# X = all the features we'll use for prediction
# y = the target we want to predict (survived)

# Select feature columns (everything except 'survived')
X = df_clean.drop('survived', axis=1)
y = df_clean['survived']

print(f"Features shape: {X.shape} (rows, columns)")
print(f"Target shape: {y.shape} (rows)")
print(f"\nFeature columns: {list(X.columns)}")

# STEP 7: Split data into training and testing sets
# --------------------------------------------------
print("\n" + "=" * 50)
print("STEP 7: Splitting Data into Training and Testing Sets")
print("=" * 50)

# Training set: Used to teach the model (80% of data)
# Testing set: Used to test the model's accuracy (20% of data)
# random_state ensures we get the same split every time

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,  # 20% for testing
    random_state=42,  # For reproducibility
    stratify=y  # Keep same proportion of survivors in both sets
)

print(f"Training set size: {len(X_train)} passengers")
print(f"Testing set size: {len(X_test)} passengers")
print(f"Proportion of survivors in training: {y_train.mean():.2%}")
print(f"Proportion of survivors in testing: {y_test.mean():.2%}")

# STEP 8: Scale numerical features
# ---------------------------------
print("\n" + "=" * 50)
print("STEP 8: Scaling Numerical Features")
print("=" * 50)

# Scaling means bringing all numbers to a similar range
# This helps the model perform better
# Example: age (0-80) and fare (0-500) are on very different scales

scaler = StandardScaler()

# Fit the scaler on training data and transform both train and test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features have been scaled to have mean=0 and standard deviation=1")
print(f"Example of first row before scaling: {X_train.iloc[0].values[:3]}")
print(f"Example of first row after scaling: {X_train_scaled[0][:3]}")

# STEP 9: Train the model
# ------------------------
print("\n" + "=" * 50)
print("STEP 9: Training the Logistic Regression Model")
print("=" * 50)

# Logistic Regression is a model used for classification
# It predicts the probability of belonging to a class (survived or not)

model = LogisticRegression(max_iter=1000, random_state=42)

# Train the model on the training data
model.fit(X_train_scaled, y_train)

print("Model training completed!")
print(f"Model coefficients (feature importance):")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"  {feature}: {coef:.4f}")

# STEP 10: Make predictions on the test set
# ------------------------------------------
print("\n" + "=" * 50)
print("STEP 10: Making Predictions")
print("=" * 50)

# Use the trained model to predict survival for the test set
y_pred = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)  # Get probabilities

# Show predictions for first 10 test passengers
print("\nPredictions for first 10 test passengers:")
print("Passenger | Actual | Predicted | Probability of Survival")
print("-" * 60)
for i in range(10):
    actual = y_test.iloc[i]
    pred = y_pred[i]
    prob = y_pred_prob[i][1]  # Probability of survival (class 1)
    print(f"    {i+1:2d}   |   {actual}    |     {pred}     |        {prob:.2%}")

# STEP 11: Evaluate Model Performance
# ------------------------------------
print("\n" + "=" * 50)
print("STEP 11: Evaluating Model Performance")
print("=" * 50)

# 11.1: Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2%}")
print("This means the model correctly predicted survival status for {:.2%} of passengers".format(accuracy))

# 11.2: Detailed Classification Report
print("\nClassification Report:")
print("-" * 60)
print("Precision: % of passengers predicted to survive who actually survived")
print("Recall: % of actual survivors that the model correctly identified")
print("F1-Score: Harmonic mean of precision and recall")
print("-" * 60)
print(classification_report(y_test, y_pred, target_names=['Not Survived', 'Survived']))

# 11.3: Confusion Matrix
print("\nConfusion Matrix:")
print("-" * 60)
print("This shows where the model made correct and incorrect predictions:")
print("              | Predicted Not Survived | Predicted Survived")
print("Actual Not Survived |      TN             |       FP")
print("Actual Survived     |      FN             |       TP")
print("-" * 60)
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix (numbers):")
print(cm)
tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Negatives (correctly predicted not survived): {tn}")
print(f"False Positives (incorrectly predicted survived): {fp}")
print(f"False Negatives (incorrectly predicted not survived): {fn}")
print(f"True Positives (correctly predicted survived): {tp}")

# STEP 12: Feature Importance Summary
# ------------------------------------
print("\n" + "=" * 50)
print("STEP 12: What Did We Learn? (Feature Importance)")
print("=" * 50)

# The coefficients tell us which features were most important
# Positive coefficient: higher value increases survival chance
# Negative coefficient: higher value decreases survival chance

# Get absolute values to see importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0],
    'Importance': np.abs(model.coef_[0])
}).sort_values('Importance', ascending=False)

print("\nFeature Importance (higher = more influential):")
print(importance.to_string(index=False))

# STEP 13: Example Predictions
# -----------------------------
print("\n" + "=" * 50)
print("STEP 13: Example Predictions for New Passengers")
print("=" * 50)

# Let's create some hypothetical passengers and predict their survival
example_passengers = pd.DataFrame([
    {'pclass': 1, 'age': 30, 'sibsp': 0, 'parch': 0, 'fare': 100, 
     'sex_encoded': 0, 'port_C': 0, 'port_Q': 0, 'port_S': 1},
    {'pclass': 3, 'age': 25, 'sibsp': 0, 'parch': 0, 'fare': 10,
     'sex_encoded': 0, 'port_C': 0, 'port_Q': 0, 'port_S': 1},
    {'pclass': 1, 'age': 40, 'sibsp': 0, 'parch': 0, 'fare': 150,
     'sex_encoded': 1, 'port_C': 0, 'port_Q': 0, 'port_S': 1},
    {'pclass': 3, 'age': 8, 'sibsp': 1, 'parch': 2, 'fare': 20,
     'sex_encoded': 1, 'port_C': 0, 'port_Q': 0, 'port_S': 1},
])

# Ensure columns are in the same order as X
example_passengers = example_passengers[X.columns]

# Scale the example passengers
example_scaled = scaler.transform(example_passengers)

# Make predictions
example_pred = model.predict(example_scaled)
example_probs = model.predict_proba(example_scaled)

print("\nPredictions for Example Passengers:")
print("-" * 70)
print("Passenger Description                    | Survived? | Probability")
print("-" * 70)

descriptions = [
    "1st class, 30yr old male",
    "3rd class, 25yr old male", 
    "1st class, 40yr old female",
    "3rd class, 8yr old female with family"
]

for i, desc in enumerate(descriptions):
    survival = "YES" if example_pred[i] == 1 else "NO"
    prob = example_probs[i][1]
    print(f"{desc:35} |   {survival}   |     {prob:.2%}")

print("\n" + "=" * 50)
print("END OF CODE - TITANIC SURVIVAL PREDICTION COMPLETE")
print("=" * 50)