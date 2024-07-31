import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
# file_path = '/Users/dakshagrawal/Documents/GitHub/TroposphericMeasurement/Raw_Data.csv'
file_path = '/Users/Parth/Documents/GitHub/TroposphericMeasurement/Raw_Data.csv'

data = pd.read_csv(file_path)

# Renaming columns for easier access
data.columns = ['Site-ID', 'Station Name', 'Latitude', 'Longitude', 'MAT', 'MAP', 
                'Transmissivity', 'Cloud Cover', 'Aridity Index']

# Selecting relevant features and target variable
features = data[['Transmissivity', 'Cloud Cover', 'Latitude', 'Longitude', 'Aridity Index']]
target = data['MAT']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initializing and training the Random Forest Regressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Evaluating the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

# Printing the evaluation metrics
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results)