import pandas as pd
from matplotlib import pyplot
import joblib
import numpy as np
import datetime
import pickle
import argparse

print("Hi ")

def string_to_timestamp(string):
    date_time_obj = datetime.datetime.strptime(string, '%Y-%m-%d %H:%M:%S')
    timestamp = date_time_obj.timestamp()
    return timestamp

class data_prep():
    def __init__(self, path):
      self.df = pd.read_csv(path, sep='\s+', header=None)

    def resampler(self,time = "5min"):
      dataframe = self.df.set_index([0])
      dataframe.index = pd.to_datetime(dataframe.index,unit = "s")
      resample = dataframe.resample(time)
      self.resampled_data = resample.sum()

    def get_resampled_with_timestamp(self):
      self.final_data = self.resampled_data.reset_index()

    def get_final_data(self):
      self.resampler()
      self.get_resampled_with_timestamp()
      return self.final_data

class to_serialised_on_off():
    def __init__(self, data):
      self.data = data
      self.min_ = 0
      self.avg_ = 0
      self.max_ = 0
      self.find_min_max(flag = 1)

    def append_to_df(self, val):
      self.data.append(val)
      self.find_min_max()

    def find_min_max(self, flag = 0):
      if(flag == 0):
        val = self.data.loc[-1][-1]
        if( val > self.max_ ):
          self.max_ = val
        elif( val < self.min_ ):
          self.min_ = val
        self.avg_ += val/self.data.shape[0]
      elif(flag):
        self.max_ = max(self.data.loc[:][1])
        self.min_ = min(self.data.loc[:][1])
        self.avg_ = np.mean(self.data.loc[:][1].values)

      self.calc_thresh()

    def calc_thresh(self):
      self.thresh = (self.max_ - self.min_)/self.avg_

    def on_off(self, target_path):
      of = []
      for i in range(self.data.shape[0]):
        if(self.data.loc[i][1] > self.thresh):
          of.append(str(self.data.loc[i][0]))
        else:
          of.append(0)
      np.save(target_path, np.array(of))
      print("File Saved at the target location")
      

def main():
    print("1 \n")
    parser = argparse.ArgumentParser(description="Start and End Times")
    parser.add_argument('--num_preds', type = int)
    parser.add_argument('--data_path')
    parser.add_argument('--target_path')
    parser.add_argument('--resample', type = bool)
    args = parser.parse_args()
    prep = data_prep(args.data_path)
    data = prep.get_final_data()
    # # else:
    # data = np.load(args.data_path)
    print("Data Prepared \n")
    obj = to_serialised_on_off(data[:args.num_preds])
    obj.on_off(args.target_path)

if __name__== "__main__":
  main()