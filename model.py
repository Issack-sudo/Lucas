# Import libraries

import pandas as pd

import numpy as np

from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import mean_squared_error, r2_score, make_scorer

import mysql.connector

# Load data

data = pd.read_csv("/content/Lucas mir spectra.csv")

# Spectral pre-treatment

data['reflectance'] = data['reflectance']/100 # Convert reflectance to a percentage

wavelengths = data['wavelength'].values # Get wavelengths

scaler = StandardScaler() # Initialise scaler

scaled_reflectance = scaler.fit_transform(data['reflectance'].values.reshape(-1, 1)) # Scale reflectance values

# 1st derivative Savitzgy-Golay, gap of 20 points

first_derivative = np.gradient(scaled_reflectance, wavelengths, axis=0) # Calculate first derivative

# Principal component analysis (PCA), use only most relevant number of scores (explaining 95% of explained variance)

pca = PCA(n_components=0.95, whiten=True) # Initialise PCA model

pca.fit(first_derivative) # Fit PCA model to data

pca_transformed = pca.transform(first_derivative) # Transform data using PCA model

# Split data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(pca_transformed, data['organic carbon'], test_size=0.2, random_state=42)

# Create and fit Support Vector Machine (SVM) model

svr = SVR() # Initialise SVR model

svr.fit(X_train, y_train) # Fit SVR model to data

# Predict values for test set

y_pred = svr.predict(X_test)

# Calculate RMSE

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Calculate R2

r2 = r2_score(y_test, y_pred)

# Calculate RPD

rpd = np.std(y_test)/rmse

# Connect to MySQL database

mydb = mysql.connector.connect(

host="localhost",

user="root",

passwd="",

database="RapidSpectraldb"

)

# Create prediction table in MySQL database

mycursor = mydb.cursor()

mycursor.execute("CREATE TABLE predictions (batch_id INT, sample_id INT, soil_property_name VARCHAR(255), prediction FLOAT)")

# Post predictions of test set to prediction table in MySQL database

predictions = pd.DataFrame({'batch_id': 1, 'sample_id': data['sample id'], 'soil_property_name': 'organic carbon', 'prediction': y_pred})

for index, row in predictions.iterrows():
  sql = "INSERT INTO predictions (batch_id, sample_id, soil_property_name, prediction) VALUES (%s, %s, %s, %s)"

  val = (row['batch_id'], row['sample_id'], row['soil_property_name'], row['prediction'])

  mycursor.execute(sql, val)

  mydb.commit()

# Post summary statistics to prediction summary table in MySQL database

summary = pd.DataFrame({'batch_id': 1, 'RMSE': rmse, 'R2': r2, 'RPD': rpd}, index=[0])

for index, row in summary.iterrows():

  sql = "INSERT INTO prediction_summary (batch_id , RMSE , R2 , RPD) VALUES (%s, %s, %s, %s)"

  val = (row['batch_id'], row['RMSE'], row['R2'], row['RPD'])

  mycursor.execute(sql, val)

  mydb.commit()

# Close MySQL connection

mycursor.close()

mydb.close()
