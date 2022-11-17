import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,r2_score,roc_auc_score,precision_score,recall_score,f1_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2
from time import time
import lightgbm as lgb
import joblib


def gen_sequence(id_df, seq_length, seq_cols):
    #241 rows x 19 columns to 50row*19col*191 items
    id_df = id_df.iloc[:,1:-1]
    seq_cols = seq_cols[1:-1]
    df_zeros=pd.DataFrame(np.zeros((seq_length-1,id_df.shape[1])),columns=id_df.columns)
    id_df=df_zeros.append(id_df,ignore_index=True)
    # print(id_df)
    data_array = id_df[seq_cols].values
    # print(data_array)
    num_elements = data_array.shape[0]
    lstm_array=[]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        # print(start," ", stop)
        lstm_array.append(data_array[start:stop, :])
    # print(np.array(lstm_array).shape)
    return np.array(lstm_array)

# function to generate labels
def gen_label(id_df, seq_length, seq_cols,label):
    # id_df = id_df.iloc[:,1:-1]
    # seq_cols = seq_cols[1:-1]
    df_zeros=pd.DataFrame(np.zeros((seq_length-1,id_df.shape[1])),columns=id_df.columns)
    id_df=df_zeros.append(id_df,ignore_index=True)
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    y_label=[]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        y_label.append(id_df[label][stop])
    return np.array(y_label)

col_name = ['EngineNo', 'Cycle']
opt_settings = ['OpSet1', 'OpSet2', 'OpSet3']
sensor_data = ['FanInletTemp', 'LPCOutletTemp', 'HPCOutletTemp', 'LPTOutletTemp', 'FanInletPressure',
              'ByPassDuctPressure', 'TotalHPCOutletPressure', 'PhysicalFanSpeed', 'PhysicalCoreSpeed',
              'EnginePressureRatio', 'StaticHPCOutletPressure', 'FuelFlowRatio', 'CorctFanSpeed', 'CorctCoreSpeed', 'BPR',
              'BurnerFuelRatio', 'BleedEnthalpy', 'DemandFanSpeed', 'DemandCorctFanSpeed', 'HPTCoolantBleed', 'LPTCoolantBleed']

df_raw = pd.read_csv('CMAPSSData/train_FD001.txt', sep='\s+', header=None, names=col_name+opt_settings+sensor_data)
df_test = pd.read_csv('CMAPSSData/test_FD001.txt', sep='\s+', header=None, names=col_name+opt_settings+sensor_data)
df_test_rul = pd.read_csv('CMAPSSData/RUL_FD001.txt', header=None, names=['RUL'])

#Train Data RUL

# From the grouped data, use the cycle coulmn and find the max
max_cycle = df_raw.groupby('EngineNo')['Cycle'].max()
df_train = df_raw.merge(max_cycle.to_frame(name='max_cycle'), left_on='EngineNo', right_index=True)
del(df_raw)

df_train_rul = df_train.copy()
df_train_rul['RUL'] = df_train_rul.max_cycle - df_train_rul.Cycle
df_train_rul.drop('max_cycle', axis=1, inplace=True)


#Test data RUL

max_cycle_test = df_test.groupby('EngineNo')['Cycle'].max()
if 'EngineNo' not in df_test_rul:
    df_test_rul.insert(0,'EngineNo',range(1,1+len(df_test_rul)))
df_test_rul_merge = df_test_rul.merge(max_cycle_test.to_frame(name='max_cycle_test'), how='inner', on='EngineNo')
new_col = df_test_rul_merge.max_cycle_test + df_test_rul_merge.RUL
if 'RULMax' not in df_test_rul_merge:
    df_test_rul_merge.insert(loc=2,column='RULMax',value=new_col)
if 'RUL' in df_test_rul_merge:
    df_test_rul_merge.drop('RUL',axis=1,inplace=True)
df_test_rul_merge.drop('max_cycle_test',axis=1,inplace=True)
df_test_with_rul = pd.merge(df_test,df_test_rul_merge, on='EngineNo')
rul_dif = df_test_with_rul.RULMax - df_test_with_rul.Cycle
if 'RUL' not in df_test_with_rul:
    df_test_with_rul['RUL'] = rul_dif
if 'RULMax' in df_test_with_rul:
    df_test_with_rul.drop('RULMax',axis=1,inplace=True)


#Feature Selection

# Removing unnecessary Columns from train and test data

df_train_cleaned = df_train_rul.filter(
    ['EngineNo', 'Cycle', 'OpSet1', 'OpSet2', 'LPCOutletTemp', 'HPCOutletTemp', 'LPTOutletTemp',
     'TotalHPCOutletPressure', 'PhysicalFanSpeed', 'PhysicalCoreSpeed', 'StaticHPCOutletPressure', 'FuelFlowRatio',
     'CorctFanSpeed', 'CorctCoreSpeed', 'BPR', 'HPTCoolantBleed', 'LPTCoolantBleed', 'RUL'], axis=1)
df_test_cleaned = df_test_with_rul.filter(
    ['EngineNo', 'Cycle', 'OpSet1', 'OpSet2', 'LPCOutletTemp', 'HPCOutletTemp', 'LPTOutletTemp',
     'TotalHPCOutletPressure', 'PhysicalFanSpeed', 'PhysicalCoreSpeed', 'StaticHPCOutletPressure', 'FuelFlowRatio',
     'CorctFanSpeed', 'CorctCoreSpeed', 'BPR', 'HPTCoolantBleed', 'LPTCoolantBleed', 'RUL'], axis=1)


#Train Data Normalization
scaler = preprocessing.MinMaxScaler()
names = df_train_cleaned.iloc[:,2:-1].columns
scaled_data = scaler.fit_transform(df_train_cleaned.iloc[:,2:-1])
df_train_scaled = pd.concat([df_train_cleaned.iloc[:,0:2], pd.DataFrame(scaled_data, columns=names),df_train_cleaned.iloc[:,-1]], axis=1)


#Test Data Normalization
# scaler = preprocessing.MinMaxScaler()
names = df_test_cleaned.iloc[:,2:-1].columns
scaled_data = scaler.transform(df_test_cleaned.iloc[:,2:-1])
df_test_scaled = pd.concat([df_test_cleaned.iloc[:,0:2], pd.DataFrame(scaled_data, columns=names),df_test_cleaned.iloc[:,-1]], axis=1)

#Saving the scaler
scaler_filename = "scaler.save"
joblib.dump(scaler, scaler_filename)

#Adding Target Label to the dataset
cycle=30
df_train_scaled['main_threshold'] = df_train_scaled['RUL'].apply(lambda x: 1 if x <= cycle else 0)
df_test_scaled['main_threshold'] = df_test_scaled['RUL'].apply(lambda x: 1 if x <= cycle else 0)

#Model Creation

X, y = df_train_scaled.drop(['RUL','EngineNo','Cycle','main_threshold'], axis=1), df_train_scaled['main_threshold']
Xt, yt = df_test_scaled.drop(['RUL','EngineNo','Cycle','main_threshold'], axis=1), df_test_scaled['main_threshold']


X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=5)
print('X_train shape : ',X_train.shape)
print('X_test shape : ',X_test.shape)
print('y_train shape : ',y_train.shape)
print('y_test shape : ',y_test.shape)

lgb_clss = lgb.LGBMClassifier(learning_rate=0.01,n_estimators=5000,num_leaves=100,objective='binary', metrics='auc',random_state=40,n_jobs=-1)
lgb_clss.fit(X_train, y_train)
lgb_clss.score(X_test, y_test)
preds2 = lgb_clss.predict(X_test)
print('Acc Score: ',accuracy_score(y_test, preds2))
print('Roc Auc Score: ',roc_auc_score(y_test, preds2))
print('Precision Score: ',precision_score(y_test, preds2))
print('Recall Score: ',recall_score(y_test, preds2))
print('f1 score: ',f1_score(y_test, preds2))

test_pred_lgb = lgb_clss.predict(Xt)
print("Test Accuracy: ",accuracy_score(yt,test_pred_lgb))

lgb_clss.booster_.save_model('lgbr_base.txt')
print(preds2)
print(np.count_nonzero(test_pred_lgb == 0),np.count_nonzero(test_pred_lgb == 1))

print('Confusion Matrix: \n',confusion_matrix(yt,test_pred_lgb))

lo_model = lgb.Booster(model_file='lgbr_base.txt')
tp2 = lo_model.predict(Xt)
tp_10 = (tp2 >= 0.5).astype('int')
# print("Test Accuracy: ",accuracy_score(yt,tp2))
print(np.count_nonzero(tp_10 == 0),np.count_nonzero(tp_10 == 1))

print('Confusion Matrix: \n',confusion_matrix(yt,tp_10))
print(tp_10)