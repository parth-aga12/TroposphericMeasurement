import pandas as ps
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

path = '/Users/dakshagrawal/Documents/GitHub/TroposphericMeasurement/Raw_Data.csv'
dat = ps.read_csv(path)
dat.columns = ['ID', 'Name', 'Latitude', 'Longitude', 'MAT', 'MAP', 'Transmissivity', 'Cloud Cover', 'Aridity Index']
parameter = dat[['Transmissivity', 'Cloud Cover','MAP']]
rparameter = dat['MAT']
xtrainer, xtester, ytrainer, ytester = train_test_split(parameter, rparameter, test_size=0.2, random_state=42)
Predictor = RandomForestRegressor(random_state=42)
Predictor.fit(xtrainer,ytrainer)
ypredicted = Predictor.predict(xtester)
AverageError = mean_absolute_error(ytester, ypredicted)
SquaredError = mean_squared_error(ytester, ypredicted)

print("Mean Average Error is:",AverageError)
print("Mean Squared Error is:",SquaredError)
results = ps.DataFrame({'Actual': ytester, 'Predicted': ypredicted})
print(results)
