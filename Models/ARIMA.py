import joblib
import numpy as np
import matplotlib.pyplot as plt
import argparse
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error,mean_absolute_error

def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')

def run_ARIMA(eq_num, src_dir, tar_path, num_preds = 1000, weights = None, draw_graph=False):
    train_x = np.load(file=src_dir+'x_eq'+str(eq_num)+'.npy')
    train_y = np.load(file=src_dir+'y_eq'+str(eq_num)+'.npy')
    test_y = np.load(file=src_dir+'xt_eq'+str(eq_num)+'.npy')
    test_x = np.load(file=src_dir+'yt_eq'+str(eq_num)+'.npy')

    history = list(test_x[0].reshape(-1))

    predictions = []
    for t in range(num_preds): 
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test_y[t][0][0]
        history.append(obs)
        # if(t%50==0):
        #     print('predicted=%f, expected=%f' % (yhat, obs))

    predictions = np.array(predictions)
    print(predictions.shape)
    print(test_y.shape)
    error = mean_squared_error(test_y[:1000].reshape(-1), predictions)
    print('Test MSE: %.3f' % error)
    print("RMSE : %.3f"%(np.sqrt(error)))
    print("MAE : %.3f"%(mean_absolute_error(test_y[:1000].reshape(-1),predictions)))

    np.save(predictions, tar_path + str(eq_num) + 'ARIMA_preds.npy')
    # plot
    plt.plot(test_y[:1000].reshape(-1), c = 'yellow')
    plt.plot(predictions, c = 'blue')
    plt.savefig(tar_path+str(eq_num)+'ARIMA_graph.png')

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description="Run ARIMA")
    parser.add_argument('--num_preds', type='int')
    parser.add_argument("--draw_graph", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="to draw graphs or not")
    parser.add_argument('--src_dir')
    parser.add_argument('--tar_dir')
    parser.add_argument('--weights')
    parser.add_argument('--eq_num', type='int')
    args = parser.parse_args()
    run_ARIMA(args.eq_num, args.src_dir, args.num_preds, args.weights, args.draw_graph, args.tar_path)

if __name__== "__main__":
  main()