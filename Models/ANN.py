import numpy as np
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Dense
import os
import argparse
from keras.models import load_model


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

class ANN:
    def __init__(self,equipment,n_members,src_path,trg_path):
        self.eq = equipment
        self.n_members = n_members
        self.s_path = src_path
        self.t_path = trg_path
        
    def fit_ann(self,x_train,y_train):
        ann = Sequential()
        ann.add(Dense(256,activation = 'relu',input_dim = 128))
        ann.add(Dense(64,activation = 'relu'))
        ann.add(Dense(64,activation = 'relu'))
        ann.add(Dense(1,activation = 'relu'))
        
        ann.compile (optimizer = 'RMSprop',loss = 'mean_squared_error',metrics = ['accuracy'])
        ann.fit(x_train,y_train, batch_size = 128, verbose = 0,epochs = 20,validation_split = 0.1)
        return ann
        
    def load_data(self):
        self.train_x = np.load(file=self.s_path+'x_eq'+str(self.eq-1)+'.npy')
        self.train_y = np.load(file==self.s_path+'xt_eq'+str(self.eq-1)+'.npy')
        self.train_x = self.train_x.reshape(-1,128)
        self.train_y = self.train_y.reshape(-1)
        test_x = np.load(file==self.s_path+'xt_eq'+str(self.eq-1)+'.npy')
        test_y = np.load(file==self.s_path+'yt_eq'+str(self.eq-1)+'.npy')
        val_split = int(0.2*test_x.shape[0])
        self.val_x = test_x[-val_split:].reshape(-1,128)
        self.val_y = test_y[-val_split:].reshape(-1)
        self.test_x = test_x[:-val_split].reshape(-1,128)
        self.test_y = test_y[:-val_split].reshape(-1)
        print(self.val_x.shape, self.val_y.shape, self.test_x.shape, self.test_y.shape)
        
    def predictions(self, n_test, idx=0):
        self.preds = []
        ann_preds = []
        for i in range(n_test):
            ann_preds.append(self.members[0].predict(self.test_x[i])
        self.preds = ann_preds
        self.metricss(n_test)
                             
    def load_all_models(self):
        all_models = list()
        for i in range(self.n_members):
            filename = self.t_path + str(self.eq-1) + '_model_' + str(i + 1) + '.h5'
            model = load_model(filename)
            all_models.append(model)
            print('>>> loaded %s' % filename)
        return all_models    
    
    def metricss(self, n_test):
        preds = self.preds[:n_test]
        test_y = self.y_test[:n_test]
        error = mean_squared_error(test_y.reshape(-1), preds)
        print('Test MSE: %.3f' % error)
        print("RMSE : %.3f"%(np.sqrt(error)))
        print("MAE : %.3f"%(mean_absolute_error(test_y.reshape(-1),preds)))
        
    def pipeline(self, n_test, train=True):
        self.load_data()
        if(train):
            for i in range(self.n_members):
                print(self.x_train.shape)
                model = self.fit_ann(self.train_x, self.train_y)
                filename = self.t_path + str(self.eq-1) + '_model_' + str(i + 1) + '.h5'
                model.save(filename)
                print('>>> Saved %s' % filename)

        self.members = self.load_all_models()
        print('Loaded %d models' % len(self.members))

        self.make_ann_comparison_graph(n_test, idx)   
        
    def make_ann_comparison_graph(self, n_test, idx=0):
        self.predictions(n_test, idx)
        plt.plot(self.test_y[idx:idx+n_test].reshape(-1), c = 'yellow', label='actual')
        plt.plot(self.preds[idx], c = 'red', label='ANN')
        plt.ylabel('Value')
        plt.xlabel('Index of Time Series Data')
        plt.legend()
        plt.savefig(self.t_path + str(self.eq-1) + '_graph_' + '.png')
        plt.show()
                             
        
    def run_ANN(num_equips, n_test, n_models, src_dir, tgt_dir, train = True):
        for i in range(num_equips):
            d["obj{0}".format(i)] = ANN_pipeline(i+1, n_models, src_dir, tgt_dir)
            d["obj{0}".format(i)].pipeline(n_test, train)
                             
    def __main__():
        parser = argparse.ArgumentParser(description="Run ANN")
        parser.add_argument('--num_equips', type='int')
        parser.add_argument('--train', type='bool')
        parser.add_argument('--src_dir')
        parser.add_argument('--tgt_dir')
    # parser.add_argument('--weights')
        parser.add_argument('--n_test', type='int')
        parser.add_argument('--n_models', type = 'int')
        args = parser.parse_args()
        run_ANN(args.num_equips, args.n_test, args.n_models, args.src_dir, args.tgt_dir, args.train)                         