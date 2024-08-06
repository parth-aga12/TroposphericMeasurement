import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# Load the dataset
file_path = '/Users/dakshagrawal/Documents/GitHub/TroposphericMeasurement/Raw_Data.csv'
data = pd.read_csv(file_path)

# Renaming columns for easier access
data.columns = ['Site-ID', 'Station Name', 'Latitude', 'Longitude', 'MAT', 'MAP', 
                'Transmissivity', 'Cloud Cover', 'Aridity Index']

# Selecting relevant features and target variable
features = data[['Latitude', 'Longitude', 'Transmissivity', 'Cloud Cover', 'Aridity Index']]
target = data['MAT']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Initialize the XGBoost Regressor
xgb = XGBRegressor(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid,
                           cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

# Fit GridSearchCV to the data
grid_search.fit(X_train, y_train)

# Get the best parameters from the grid search
best_params = grid_search.best_params_

# Train the XGBoost Regressor with the best parameters
best_xgb = XGBRegressor(**best_params, random_state=42)
best_xgb.fit(X_train, y_train)

# Make predictions on the test set
y_pred_best = best_xgb.predict(X_test)

# Evaluate the model with the best parameters
mae_best = mean_absolute_error(y_test, y_pred_best)
rmse_best = mean_squared_error(y_test, y_pred_best) ** 0.5

# Print the evaluation metrics and the best parameters
print(f'Best Parameters: {best_params}')
print(f'Mean Absolute Error (MAE) with best parameters: {mae_best}')
print(f'Root Mean Squared Error (RMSE) with best parameters: {rmse_best}')

# Display the actual vs predicted results
results_best = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_best})
print(results_best)