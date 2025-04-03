import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

# Load dataset
data = pd.read_csv('houses.csv')

# Remove outliers based on price
Q1 = data['price'].quantile(0.25)
Q3 = data['price'].quantile(0.75)
IQR = Q3 - Q1
data = data[(data['price'] >= Q1 - 1.5 * IQR) & (data['price'] <= Q3 + 1.5 * IQR)]

# Feature engineering
data['price_per_sqft'] = data['price'] / data['sqft_lot']

# Select features and target
X = data[['bedrooms', 'sqft_lot', 'city', 'price_per_sqft']]
y = np.log1p(data['price'])  # Log transformation

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
numeric_features = ['bedrooms', 'sqft_lot', 'price_per_sqft']
numeric_transformer = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler())
])

categorical_features = ['city']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Define model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.05, 0.1]
}
random_search = RandomizedSearchCV(xgb_model, param_grid, n_iter=15, cv=5, scoring='neg_mean_absolute_error', random_state=42)

# Pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', random_search)
])

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = np.expm1(model.predict(X_test))  # Reverse log transformation

# Evaluate
mae = mean_absolute_error(np.expm1(y_test), y_pred)
mse = mean_squared_error(np.expm1(y_test), y_pred)
rmse = np.sqrt(mse)

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')

