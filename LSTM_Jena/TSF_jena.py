import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from WindowGenerator import WindowGenerator


train_df = pd.read_csv('preprocessed_normalized_train_df_70.csv')
val_df = pd.read_csv('preprocessed_normalized_validation_df_20.csv')
test_df = pd.read_csv('preprocessed_normalized_test_df_10.csv')

train_df.drop('Unnamed: 0', axis=1, inplace=True)
val_df.drop('Unnamed: 0', axis=1, inplace=True)
test_df.drop('Unnamed: 0', axis=1, inplace=True)

print(train_df.shape,val_df.shape,test_df.shape)
print(train_df.columns)


column_indices = {name: i for i, name in enumerate(train_df.columns)}
num_features = train_df.shape[1]
val_performance = {}
performance = {}

#single_step_window
single_step_window = WindowGenerator(
    input_width=1, label_width=1, shift=1,
    train_df=train_df, val_df=val_df, test_df=test_df,
    label_columns=['T (degC)'])

print("single_step_window ")

# Wide Window Model
wide_window = WindowGenerator(
    input_width=24, label_width=24, shift=1,
    train_df=train_df, val_df=val_df, test_df=test_df,
    label_columns=['T (degC)'])

print("wide_window ")


MAX_EPOCHS = 20

def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsolutePercentageError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping], verbose=0)
  return history


# Baseline Model

class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]

baseline = Baseline(label_index=column_indices['T (degC)'])
baseline.compile(loss=tf.losses.MeanSquaredError(),
                 metrics=[tf.metrics.MeanAbsolutePercentageError()])
val_performance['Baseline'] = baseline.evaluate(single_step_window.val, verbose=2)
performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=2)

print('Baseline:',val_performance['Baseline'])
print('Baseline:',performance['Baseline'])
print(wide_window)

# LSTM Model

lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dense(units=1)
])
history = compile_and_fit(lstm_model, wide_window)
val_performance['LSTM'] = lstm_model.evaluate(wide_window.val, verbose=2)
performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=2)

print('LSTM Val Performance:',val_performance['LSTM'])
print('LSTM Test Performance:',performance['LSTM'])

'''
# GRU
gru = tf.keras.models.Sequential([
    tf.keras.layers.GRU(32, return_sequences=True),
    tf.keras.layers.Dense(units=1)
])
history = compile_and_fit(gru, wide_window)
val_performance['GRU'] =gru.evaluate(wide_window.val, verbose=2)
performance['GRU'] = gru.evaluate(wide_window.test, verbose=2)

print('GRU Val Performance:',val_performance['GRU'])
print('GRU Test Performance:',performance['GRU'])



# Coupled Input and Foreget Gate 
from LSTM_CIFG import LSTM_CIFG
CIFG = tf.keras.models.Sequential([
    tf.keras.layers.RNN(LSTM_CIFG(32), return_sequences=True),
    tf.keras.layers.Dense(units=1)
])
history = compile_and_fit(CIFG, wide_window)
val_performance['LSTM_CIFG'] = CIFG.evaluate(wide_window.val, verbose=2)
performance['LSTM_CIFG'] = CIFG.evaluate(wide_window.test, verbose=2)

print('CIFG:',val_performance['LSTM_CIFG'])
print('CIFG',performance['LSTM_CIFG'])




# No input Gate
from LSTM_NIG import LSTM_NIG
NIG = tf.keras.models.Sequential([
    tf.keras.layers.RNN(LSTM_NIG(32), return_sequences=True),
    tf.keras.layers.Dense(units=1)
])
history = compile_and_fit(NIG, wide_window)
val_performance['LSTM_NIG'] = NIG.evaluate(wide_window.val, verbose=2)
performance['LSTM_NIG'] = NIG.evaluate(wide_window.test, verbose=2)

print('NIG:',val_performance['LSTM_NIG'])
print('NIG',performance['LSTM_NIG'])


# No Forget Gate
from LSTM_NFG import LSTM_NFG
NFG = tf.keras.models.Sequential([
    tf.keras.layers.RNN(LSTM_NFG(32), return_sequences=True),
    tf.keras.layers.Dense(units=1)
])
history = compile_and_fit(NFG, wide_window)
val_performance['LSTM_NFG'] = NFG.evaluate(wide_window.val, verbose=2)
performance['LSTM_NFG'] = NFG.evaluate(wide_window.test, verbose=2)

print('NFG',val_performance['LSTM_NFG'])
print('NFG',performance['LSTM_NFG'])


# No Output Gate
from LSTM_NOG import LSTM_NOG
NOG = tf.keras.models.Sequential([
    tf.keras.layers.RNN(LSTM_NOG(32), return_sequences=True),
    tf.keras.layers.Dense(units=1)
])
history = compile_and_fit(NOG, wide_window)
val_performance['LSTM_NOG'] = NOG.evaluate(wide_window.val, verbose=2)
performance['LSTM_NOG'] = NOG.evaluate(wide_window.test, verbose=2)

print('NOG',val_performance['LSTM_NOG'])
print('NOG',performance['LSTM_NOG'])



#NIA No input activation
from LSTM_NIA import LSTM_NIA
NIAG = tf.keras.models.Sequential([
    tf.keras.layers.RNN(LSTM_NIA(32), return_sequences=True),
    tf.keras.layers.Dense(units=1)
])
history = compile_and_fit(NIAG, wide_window)
val_performance['LSTM_NIA'] = NIAG.evaluate(wide_window.val, verbose=2)
performance['LSTM_NIA'] = NIAG.evaluate(wide_window.test, verbose=2)

print('NIAG',val_performance['LSTM_NIA'])
print('NIAG',performance['LSTM_NIA'])


#NOA No input activation
from LSTM_NOA import LSTM_NOA
NOA = tf.keras.models.Sequential([
    tf.keras.layers.RNN(LSTM_NOA(32), return_sequences=True),
    tf.keras.layers.Dense(units=1)
])
history = compile_and_fit(NOA, wide_window)
val_performance['LSTM_NOA'] = NOA.evaluate(wide_window.val, verbose=2)
performance['LSTM_NOA'] = NOA.evaluate(wide_window.test, verbose=2)

print('NOA',val_performance['LSTM_NOA'])
print('NOA',performance['LSTM_NOA'])
'''

# Peephole Connections
PeepholeLSTM = tf.keras.models.Sequential([
    tf.keras.layers.RNN(tf.keras.experimental.PeepholeLSTMCell(32), return_sequences=True),
    tf.keras.layers.Dense(units=1)
])

history = compile_and_fit(PeepholeLSTM, wide_window)
val_performance['PeepholeLSTM'] = PeepholeLSTM.evaluate(wide_window.val, verbose=2)
performance['PeepholeLSTM'] = PeepholeLSTM.evaluate(wide_window.test, verbose=2)

print('PeepholeLSTM',val_performance['PeepholeLSTM'])
print('PeepholeLSTM',performance['PeepholeLSTM'])



######### Download Result

val_results_df = pd.DataFrame.from_dict(val_performance)
test_results_df = pd.DataFrame.from_dict(performance)

val_results_df.to_csv("val_results_df.cvs")
test_results_df.to_csv("test_results_df.csv")

######## Plot Results
x = np.arange(len(performance))
width = 0.3
metric_name = 'mean_absolute_percentage_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_percentage_error')
val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]

plt.ylabel('mean_absolute_percentage_error [T (degC), normalized]')
plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=performance.keys(),
           rotation=45)
_ = plt.legend()
plt.show()


