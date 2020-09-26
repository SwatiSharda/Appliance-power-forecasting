import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import datetime
import argparse

def generate_data_files(path_s, s_name, path_t, num_equips, n_steps = 128, resample = False):
    # if(resample):
    #     dat_to_dataframe(path_s+s_name+".pkl")

    for i in range(num_equips):
        data = joblib.load(path_s+s_name+str(i+1)+".pkl")
        m = data.shape[0]
        n = data.shape[1]
        data = data.values
        data = data.reshape((-1,1))
        x_final = []
        y_final  = []
        for j in range(m):
            if j<n_steps:
                continue
            x_temp = data[j-n_steps:j,0:1]
            y_temp = data[j,0:1]    
            x_final.append(x_temp)
            y_final.append(y_temp)
            
        x_final = np.array(x_final)
        y_final = np.array(y_final)
        y_final = y_final.reshape((-1,))
        x_final = x_final.reshape((-1,n_steps,1))

        print(x_final.shape)
        print(y_final.shape)

        x_train,x_test,y_train,y_test = train_test_split(x_final,y_final,test_size = 0.3)

        np.save(path_t+"x_eq"+str(i)+".npy", x_train)
        np.save(path_t+"y_eq"+str(i)+".npy", y_train)
        np.save(path_t+"xt_eq"+str(i)+".npy", x_test)
        np.save(path_t+"yt_eq"+str(i)+".npy", y_test)

def __main__() :
    parser = argparse.ArgumentParser(description = 'Prepare the Dataset')
    parser.add_argument('--resample', type = 'bool')
    parser.add_argument('--num_equips', type = 'int')
    parser.add_argument('--src_dir')
    parser.add_argument('--src_file_name')
    parser.add_argument('--tar_dir')
    parser.add_argument('--n_steps', type='int')
    args=parser.parse_args()
    generate_data_files(args.src_dir, args.src_file_name, args.tar_dir, args.num_equips, args.n_steps, args.resample)