import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

# Load the dataset
# file_path = 'C:/Users/Parth/Documents/GitHub/TroposphericMeasurement/Raw_Data.csv'
# data = pd.read_csv(file_path)

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"
# currently set to toronto (maybe add feature to choose location which then changes longitude latitude)
params = {
	"latitude": 43.7001,
	"longitude": -79.4163,
	"start_date": "2024-07-19",
	"end_date": "2024-08-02",
	"hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "surface_pressure", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "is_day"]
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation {response.Elevation()} m asl")
print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()
hourly_surface_pressure = hourly.Variables(3).ValuesAsNumpy()
hourly_cloud_cover = hourly.Variables(4).ValuesAsNumpy()
hourly_cloud_cover_low = hourly.Variables(5).ValuesAsNumpy()
hourly_cloud_cover_mid = hourly.Variables(6).ValuesAsNumpy()
hourly_cloud_cover_high = hourly.Variables(7).ValuesAsNumpy()
hourly_is_day = hourly.Variables(8).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}
hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
hourly_data["precipitation"] = hourly_precipitation
hourly_data["surface_pressure"] = hourly_surface_pressure
hourly_data["cloud_cover"] = hourly_cloud_cover
hourly_data["cloud_cover_low"] = hourly_cloud_cover_low
hourly_data["cloud_cover_mid"] = hourly_cloud_cover_mid
hourly_data["cloud_cover_high"] = hourly_cloud_cover_high
hourly_data["is_day"] = hourly_is_day

data = pd.DataFrame(data = hourly_data)

# Renaming columns for easier access
data.columns = ['Time', 'Temperature', 'Relative Humidity', 'Precipitation',
                'Surface Pressure', 'Cloud Cover', 'Cloud Cover Low', 'Cloud Cover Mid', 'Cloud Cover High', 'Is Day']

# Selecting relevant features and target variable
features = data[['Cloud Cover', 'Cloud Cover Low', 'Cloud Cover Mid', 'Cloud Cover High', 'Relative Humidity', 'Precipitation', 'Is Day', 'Surface Pressure']]
target = data['Temperature']

# Splitting the data into training and testing sets]
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

"""
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
"""

# Best parameters from the grid search
best_params = {
    'n_estimators': 100,
    'learning_rate': 0.05,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.9
}

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

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual MAT')
plt.ylabel('Predicted MAT')
plt.title('Actual vs Predicted Mean Annual Temperature')
plt.show()