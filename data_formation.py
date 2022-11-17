import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
import numpy as np
import lightgbm
import joblib

df = pd.read_csv("Data_sheet.csv")
df_filtered = df.filter(['EngineNo', 'Cycle', 'OpSet1', 'OpSet2', 'LPCOutletTemp', 'HPCOutletTemp', 'LPTOutletTemp',
                         'TotalHPCOutletPressure', 'PhysicalFanSpeed', 'PhysicalCoreSpeed', 'StaticHPCOutletPressure',
                         'FuelFlowRatio', 'CorctFanSpeed', 'CorctCoreSpeed', 'BPR', 'HPTCoolantBleed',
                         'LPTCoolantBleed'], axis=1)
# print(df_filtered)
scaler = joblib.load("scaler.save")
# scaler = preprocessing.MinMaxScaler()
names = df_filtered.iloc[:,2:].columns
scaled_data = scaler.transform(df_filtered.iloc[:,2:])
df_test_scaled = pd.concat([df_filtered.iloc[:,0:2], pd.DataFrame(scaled_data, columns=names)], axis=1)
# print(df_test_scaled)
model = lightgbm.Booster(model_file='lgbr_base.txt')


X_pred = df_test_scaled.drop(['EngineNo','Cycle'], axis=1)
y_pred = model.predict(X_pred)

y_pred_lstm = (y_pred >= 0.5).astype('int')
print(y_pred_lstm)
print(y_pred_lstm.shape)
print(np.count_nonzero(y_pred_lstm == 0),np.count_nonzero(y_pred_lstm == 1))






