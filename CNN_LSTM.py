import joblib
from keras.layers import Dense ,LSTM
from keras.models import Sequential 
from sklearn.metrics import mean_absolute_error,mean_squared_error
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import pyplot
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Flatten
from sklearn.metrics import r2_score as r2Conversations
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K

import os
import argparse
from keras.models import load_model

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

class CNN_LSTM_pipeline():
  def __init__(self, equipment, n_members, src_path, tgt_path):
    self.eq = equipment
    self.n_members = n_members
    self.s_path = src_path
    self.t_path = tgt_path

  # def load_arima_res(self):
  #   return np.load('./drive/My Drive/UkDale/ARIMA/arima_ukdale_preds_1000_eq'+str(self.eq)+'.npy')

  def fit_cnn_lstm(self, x_train, y_train):
    n_timesteps, n_features = x_train.shape[1], x_train.shape[2]
    print(n_timesteps)
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,1)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(RepeatVector(1))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    # model.summary()
    model.compile(loss='mse', optimizer='adam',metrics = ["accuracy"])

    model.fit(x_train, y_train.reshape(-1,1,1), epochs=10, batch_size=128, verbose = 1)
    
    return model

  def load_data(self):
    self.train_x = np.load(file=self.s_path+'x_eq'+str(self.eq-1)+'.npy')
    self.train_y = np.load(file=self.s_path+'xt_eq'+str(self.eq-1)+'.npy')
    test_x = np.load(file=self.s_path+'xt_eq'+str(self.eq-1)+'.npy')
    test_y = np.load(file=self.s_path+'yt_eq'+str(self.eq-1)+'.npy')
    val_split = int(0.2*test_x.shape[0])
    self.val_x = test_x[-val_split:]
    self.val_y = test_y[-val_split:]
    self.test_x = test_x[:-val_split]
    self.test_y = test_y[:-val_split]
    print(self.val_x.shape, self.val_y.shape, self.test_x.shape, self.test_y.shape)
  
  def load_all_models(self):
    all_models = list()
    for i in range(self.n_members):
        filename = self.t_path + str(self.eq-1) + '_model_' + str(i + 1) + '.h5'
        model = load_model(filename)
        all_models.append(model)
        print('>>> loaded %s' % filename)
    return all_models

  def predictions(self, n_test, idx=0):
    self.preds = []
    lstm_preds = []
    for i in range(n_test):
      lstm_preds.append(self.members[0].predict(self.test_x[i].reshape(1,-1,1))[0][0][0])
    self.preds.append(lstm_preds)
    self.metricss(n_test)

  def make_cnnlstm_comparison_graph(self, n_test, idx=0):
    self.predictions(n_test, idx)
    plt.plot(self.test_y[idx:idx+n_test].reshape(-1), c = 'yellow', label='actual')
    plt.plot(self.preds[j], c = 'red', label='CNN-LSTM')
    plt.ylabel('Value')
    plt.xlabel('Index of Time Series Data')
    plt.legend()
    plt.savefig(self.t_path + str(self.eq-1) + '_graph_' + '.png')
    plt.show()

  def metricss(self, n_test):
    preds = self.preds[:n_test]
    test_y = self.test_y[:n_test]
    error = mean_squared_error(test_y.reshape(-1), preds)
    print('Test MSE: %.3f' % error)
    print("RMSE : %.3f"%(np.sqrt(error)))
    print("MAE : %.3f"%(mean_absolute_error(test_y.reshape(-1),preds)))

  def pipeline(self, n_test, train=True):
    self.load_data()
    if(train):
        for i in range(self.n_members):
          print(self.train_x.shape)
          model = self.fit_cnn_lstm(self.train_x, self.train_y)
          filename = self.t_path + str(self.eq-1) + '_model_' + str(i + 1) + '.h5'
          model.save(filename)
          print('>>> Saved %s' % filename)

    self.members = self.load_all_models()
    print('Loaded %d models' % len(self.members))

    self.make_cnnlstm_comparison_graph(n_test, idx)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def run_CNN_LSTM(num_equips, n_test, n_models, src_dir, tgt_dir, train = True):
    d = {}
    for i in range(num_equips):
        d["obj{0}".format(i)] = CNN_LSTM_pipeline(i+1, n_models, src_dir, tgt_dir)
        d["obj{0}".format(i)].pipeline(n_test, train)

def main():
    parser = argparse.ArgumentParser(description="Run CNN_LSTM")
    parser.add_argument('--num_equips', type='int')
    parser.add_argument("--train", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="to train or not")
    parser.add_argument('--src_dir')
    parser.add_argument('--tgt_dir')
    # parser.add_argument('--weights')
    parser.add_argument('--n_test', type='int')
    parser.add_argument('--n_models', type = 'int')
    args = parser.parse_args()
    run_CNN_LSTM(args.num_equips, args.n_test, args.n_models, args.src_dir, args.tgt_dir, args.train)

if __name__== "__main__":
  main()