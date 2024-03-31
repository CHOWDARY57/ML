import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

file_path = "C:/Users/ajayk/OneDrive/Documents/car data.csv"
data = pd.read_csv(file_path)
data.dropna(inplace=True)
X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']
categorical_cols = ['Car_Name', 'Fuel_Type', 'Seller_Type', 'Transmission']
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_cols)], remainder='passthrough')
X_encoded = ct.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
