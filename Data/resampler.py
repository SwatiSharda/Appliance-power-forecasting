import pandas as pd
from matplotlib import pyplot
from sklearn.externals import joblib
import numpy as np
 
file = open("channel_15.dat")
data = file.readlines()
rnn_data=[]

for i in data:
    temp = i.split(" ")
    rnn_data.append([int(x) for x in temp])
            
rnn_data = pd.DataFrame(rnn_data)
rnn_data = rnn_data.set_index([0])
rnn_data.index = pd.to_datetime(rnn_data.index,unit = "s")
resampler_5_min = rnn_data.resample("5min")
_5min_resampled_data = resampler_5_min.sum()
_5min_resampled_data = np.array(_5min_resampled_data)
_5min_resampled_data = _5min_resampled_data/300
print(_5min_resampled_data)
joblib.dump(_5min_resampled_data,"ukdale_equip_15_house1.pkl")