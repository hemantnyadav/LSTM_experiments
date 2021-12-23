#Import
import pandas as pd
import numpy as np

#Read Dataset
df = pd.read_csv("jena_climate_2009_2016.csv")

#Convert from 10 Minits to 1 Hour, do 
#Slice [start:stop:step], starting from index 5 take every 6th record.
df = df[5::6]

# Extract Datetime and convert to Date format
date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

#Inspect and Cleanup
# Wind Velocity amd Max wind velicity is having some values 
# like -9999 So replace them with 0
df[df['wv (m/s)']==-9999.0] = 0
df[df['max. wv (m/s)']==-9999.0] = 0

#Feature Engineering
# Convert Wind Velocity and Wind Direction columns in to one

# Calculate the wind x and y components.
df['Wx'] = df['wv (m/s)']*np.cos(df['wd (deg)']*np.pi / 180)
df['Wy'] = df['wv (m/s)']*np.sin(df['wd (deg)']*np.pi / 180)

# Calculate the max wind x and y components.
df['max Wx'] = df['max. wv (m/s)']*np.cos(df['wd (deg)']*np.pi / 180)
df['max Wy'] = df['max. wv (m/s)']*np.sin(df['wd (deg)']*np.pi / 180)

df.drop(['wv (m/s)','max. wv (m/s)','wd (deg)'])
# Convert Date time column from string to seconds
timestamp_s = date_time.map(pd.Timestamp.timestamp)

day 	= 24*60*60
year 	= (365.2425)*day

df['Day sin'] 	= np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] 	= np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] 	= np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] 	= np.cos(timestamp_s * (2 * np.pi / year))

#Split the data
# (70%, 20%, 10%) split for the training, validation, and test sets. 
# Note: the data is not being randomly shuffled before splitting. This is for two reasons:
# 	It ensures that chopping the data into windows of consecutive samples is still possible.
# 	It ensures that the validation/test results are more realistic, 
#		being evaluated on the data collected after the model was trained.

column_indices = {name: i for i, name in enumerate(df.columns)}

n 		 = len(df)
train_df = df[0:int(n*0.7)]
val_df   = df[int(n*0.7):int(n*0.9)]
test_df  = df[int(n*0.9):]

# Normalize the data

train_mean = train_df.mean()
train_std  = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df   = (val_df - train_mean) / train_std
test_df  = (test_df - train_mean) / train_std

train_df.to_csv("preprocessed_normalized_train_df_70.csv")
val_df.to_csv("preprocessed_normalized_validation_df_20.csv")
test_df.to_csv("preprocessed_normalized_test_df_10.csv")