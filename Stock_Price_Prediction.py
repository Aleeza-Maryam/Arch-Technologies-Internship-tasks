"""
====================================================================
STOCK PRICE PREDICTION - COMPLETE IMPLEMENTATION (FIXED)
====================================================================
Project: Time Series Forecasting
Author: [Your Name]
Date: [Current Date]
Description: Predicts future stock prices using historical data
====================================================================
"""

# ============================================
# SECTION 1: IMPORT LIBRARIES
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

print("=" * 60)
print("STOCK PRICE PREDICTION - V1.0")
print("=" * 60)

# ============================================
# SECTION 2: DATA LOADING
# ============================================

print("\n[Step 1] Loading Stock Data...")

# Create sample data (for demonstration)
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=500, freq='D')

# Generate realistic stock prices (random walk with upward trend)
base_price = 100
trend = np.linspace(0, 20, 500)  # Gradual upward trend
noise = np.random.randn(500) * 2   # Random fluctuations

close_prices = base_price + trend + np.cumsum(noise)  # Cumulative sum for realistic movement

# Create other price columns based on close price
high_prices = close_prices + np.random.rand(500) * 3
low_prices = close_prices - np.random.rand(500) * 3
open_prices = close_prices - np.random.randn(500) * 2
volume = np.random.randint(1000000, 10000000, 500)

# Create DataFrame
df = pd.DataFrame({
    'Date': dates,
    'Open': open_prices,
    'High': high_prices,
    'Low': low_prices,
    'Close': close_prices,
    'Volume': volume
})

df.set_index('Date', inplace=True)

print(f"Dataset loaded successfully!")
print(f"   Total trading days: {len(df)}")
print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
print(f"\nFirst 5 rows of data:")
print(df.head())

# ============================================
# SECTION 3: DATA EXPLORATION
# ============================================

print("\n[Step 2] Data Exploration...")

print("\nBasic Statistics:")
print(df.describe())

# Check for missing values
print(f"\nMissing values: {df.isnull().sum().sum()}")

# ============================================
# SECTION 4: FEATURE ENGINEERING
# ============================================

print("\n[Step 3] Feature Engineering...")

# Create target variable: next day's closing price
df['Target_Close'] = df['Close'].shift(-1)

# Create additional features for better prediction
df['Price_Range'] = df['High'] - df['Low']  # Daily price range
df['Price_Change'] = df['Close'] - df['Open']  # Daily price change
df['Return'] = df['Close'].pct_change() * 100  # Daily percentage return
df['Volume_Change'] = df['Volume'].pct_change() * 100  # Volume percentage change

# Add moving averages as features
df['MA_5'] = df['Close'].rolling(window=5).mean()  # 5-day moving average
df['MA_10'] = df['Close'].rolling(window=10).mean()  # 10-day moving average
df['MA_20'] = df['Close'].rolling(window=20).mean()  # 20-day moving average

# Add volatility (standard deviation of returns)
df['Volatility'] = df['Return'].rolling(window=10).std()

# Drop NaN values created by rolling calculations and shifting
df = df.dropna()

print(f"Created {len(df.columns)} features for prediction")
print(f"Features: {list(df.columns)}")

# ============================================
# SECTION 5: PREPARE FEATURES AND TARGET
# ============================================

print("\n[Step 4] Preparing Features and Target...")

# Define features (X) and target (y)
feature_columns = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'Price_Range', 'Price_Change', 'Return', 'Volume_Change',
    'MA_5', 'MA_10', 'MA_20', 'Volatility'
]

X = df[feature_columns]
y = df['Target_Close']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# ============================================
# SECTION 6: SPLIT DATA (TIME SERIES AWARE)
# ============================================

print("\n[Step 5] Splitting Data...")

# For time series, we must NOT shuffle the data
# Use chronological order: older data for training, newer for testing

split_index = int(len(X) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

print(f"Training set: {len(X_train)} days ({df.index[0].date()} to {df.index[split_index-1].date()})")
print(f"Testing set: {len(X_test)} days ({df.index[split_index].date()} to {df.index[-1].date()})")

# ============================================
# SECTION 7: FEATURE SCALING
# ============================================

print("\n[Step 6] Scaling Features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled to mean=0, std=1")

# ============================================
# SECTION 8: TRAIN LINEAR REGRESSION MODEL
# ============================================

print("\n[Step 7] Training Linear Regression Model...")

# Model 1: Linear Regression
model_lr = LinearRegression()
model_lr.fit(X_train_scaled, y_train)

print("Linear Regression model trained!")

# Display feature importance for linear regression
print("\nFeature Coefficients (Linear Regression):")
coef_df = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': model_lr.coef_
}).sort_values('Coefficient', ascending=False)

print(coef_df.to_string(index=False))

# ============================================
# SECTION 9: TRAIN RANDOM FOREST MODEL
# ============================================

print("\n[Step 8] Training Random Forest Model...")

# Model 2: Random Forest (better for non-linear patterns)
model_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_rf.fit(X_train_scaled, y_train)

print("Random Forest model trained!")

# Display feature importance for random forest
print("\nFeature Importance (Random Forest):")
importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': model_rf.feature_importances_
}).sort_values('Importance', ascending=False)

print(importance_df.to_string(index=False))

# ============================================
# SECTION 10: MAKE PREDICTIONS
# ============================================

print("\n[Step 9] Making Predictions...")

# Predictions from both models
y_pred_lr = model_lr.predict(X_test_scaled)
y_pred_rf = model_rf.predict(X_test_scaled)

# Display sample predictions
print("\nSample Predictions (first 10 test days):")
print("-" * 80)
print("Date       | Actual Price | LR Prediction | RF Prediction")
print("-" * 80)

test_dates = df.index[split_index:split_index+10]
for i in range(10):
    date_str = test_dates[i].strftime('%Y-%m-%d')
    actual = y_test.iloc[i]
    pred_lr = y_pred_lr[i]
    pred_rf = y_pred_rf[i]
    print(f"{date_str} |     {actual:.2f}    |     {pred_lr:.2f}    |     {pred_rf:.2f}")

# ============================================
# SECTION 11: MODEL EVALUATION
# ============================================

print("\n" + "=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)

# Function to evaluate model
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"\n{model_name} Performance:")
    print("-" * 50)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R2) Score: {r2:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}

# Evaluate both models
print("\n" + "=" * 50)
print("LINEAR REGRESSION RESULTS")
print("=" * 50)
results_lr = evaluate_model(y_test, y_pred_lr, "Linear Regression")

print("\n" + "=" * 50)
print("RANDOM FOREST RESULTS")
print("=" * 50)
results_rf = evaluate_model(y_test, y_pred_rf, "Random Forest")

# ============================================
# SECTION 12: VISUALIZATION (FIXED)
# ============================================

print("\n[Step 10] Creating Visualization...")

# Get the full test dates for plotting
full_test_dates = df.index[split_index:]

# Ensure all arrays have the same length
print(f"Test dates length: {len(full_test_dates)}")
print(f"y_test length: {len(y_test)}")
print(f"y_pred_lr length: {len(y_pred_lr)}")
print(f"y_pred_rf length: {len(y_pred_rf)}")

# Create plots for visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Actual vs Predicted (Linear Regression)
ax1 = axes[0, 0]
ax1.plot(full_test_dates, y_test, label='Actual Price', linewidth=2, color='blue')
ax1.plot(full_test_dates, y_pred_lr, label='LR Prediction', linestyle='--', linewidth=2, color='green')
ax1.set_title('Linear Regression Predictions', fontsize=12)
ax1.set_xlabel('Date')
ax1.set_ylabel('Stock Price ($)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Actual vs Predicted (Random Forest)
ax2 = axes[0, 1]
ax2.plot(full_test_dates, y_test, label='Actual Price', linewidth=2, color='blue')
ax2.plot(full_test_dates, y_pred_rf, label='RF Prediction', linestyle='--', linewidth=2, color='red')
ax2.set_title('Random Forest Predictions', fontsize=12)
ax2.set_xlabel('Date')
ax2.set_ylabel('Stock Price ($)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Prediction Errors (Linear Regression)
ax3 = axes[1, 0]
errors_lr = y_test.values - y_pred_lr
ax3.plot(full_test_dates, errors_lr, color='red', linewidth=1)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.set_title('Prediction Errors - Linear Regression', fontsize=12)
ax3.set_xlabel('Date')
ax3.set_ylabel('Error ($)')
ax3.grid(True, alpha=0.3)

# Plot 4: Prediction Errors (Random Forest)
ax4 = axes[1, 1]
errors_rf = y_test.values - y_pred_rf
ax4.plot(full_test_dates, errors_rf, color='blue', linewidth=1)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax4.set_title('Prediction Errors - Random Forest', fontsize=12)
ax4.set_xlabel('Date')
ax4.set_ylabel('Error ($)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('stock_prediction_results.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'stock_prediction_results.png'")

# ============================================
# SECTION 13: COMPARISON SUMMARY
# ============================================

print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)

comparison_df = pd.DataFrame({
    'Metric': ['MSE', 'RMSE', 'MAE', 'R2', 'MAPE (%)'],
    'Linear Regression': [
        f"{results_lr['MSE']:.4f}",
        f"{results_lr['RMSE']:.4f}",
        f"{results_lr['MAE']:.4f}",
        f"{results_lr['R2']:.4f}",
        f"{results_lr['MAPE']:.2f}"
    ],
    'Random Forest': [
        f"{results_rf['MSE']:.4f}",
        f"{results_rf['RMSE']:.4f}",
        f"{results_rf['MAE']:.4f}",
        f"{results_rf['R2']:.4f}",
        f"{results_rf['MAPE']:.2f}"
    ]
})

print("\nModel Performance Comparison:")
print(comparison_df.to_string(index=False))

# Determine best model
if results_rf['RMSE'] < results_lr['RMSE']:
    best_model = "Random Forest"
    best_rmse = results_rf['RMSE']
else:
    best_model = "Linear Regression"
    best_rmse = results_lr['RMSE']

print(f"\nBest Model: {best_model} (RMSE: {best_rmse:.4f})")

# ============================================
# SECTION 14: EXAMPLE PREDICTION
# ============================================

print("\n" + "=" * 60)
print("EXAMPLE: PREDICTING NEXT DAY'S PRICE")
print("=" * 60)

# Get the most recent data point
last_data = X.iloc[-1:].copy()
last_data_scaled = scaler.transform(last_data)

# Predict next day's price using best model
if best_model == "Random Forest":
    next_price = model_rf.predict(last_data_scaled)[0]
else:
    next_price = model_lr.predict(last_data_scaled)[0]

last_actual_price = df['Close'].iloc[-1]
price_change = next_price - last_actual_price
price_change_pct = (price_change / last_actual_price) * 100

print(f"Last Actual Price: ${last_actual_price:.2f}")
print(f"Predicted Next Price: ${next_price:.2f}")
print(f"Predicted Change: ${price_change:.2f} ({price_change_pct:.2f}%)")

if price_change > 0:
    print("Prediction: Price will go UP")
else:
    print("Prediction: Price will go DOWN")

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)

# Optional: Show the plot
plt.show()