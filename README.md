# PowerForecasting
Time Series Analysis and User Behaviour Learning to forecast power usage in a household. 

## Dataset
The Dataset used to test the performance of various models for the task at hand are [GreenD](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=3&cad=rja&uact=8&ved=2ahUKEwjos-j4x6joAhXCW3wKHZkEDoUQFjACegQIAhAB&url=https%3A%2F%2Farxiv.org%2Fabs%2F1405.3100&usg=AOvVaw2-YXk54TvD_YqliJYqBUC8) and [UkDale](https://jack-kelly.com/data/). The resampled files of these datasets are present in the Data folder.
The models can be used for any new data, provided they are available in the given format. 

## Using 
### Initialisation
- Clone the repository
```
git clone https://github.com/neilgautam/PowerForecasting.git
```
- Install the dependencies 
```
pip install --r requirements.txt
```
### Prepare Data
- Preprocess the data to equipment-wise usable format 
```
python Data/preprocess.py --src_dir <path-to-directory-containing-data> --src_file_name \
       <common-prefix-for-all-equipments> --tar_dir <target-directory-to-place-preprocessed-data>\
       --num_equips <number-of-equipments> --n_steps <number-of-timesteps, default=128> 
```
### Run the Models
- ARIMA 
```
python Models/ARIMA.py --src_dir <path-to-directory-containing-preprocessed-data> --tar_dir \
      <path-where-the-outputs-need-to-be-placed> --eq_num <equipment-number-to-be-tested> \
      --num_preds <number-of-predictions-to-be-made> --draw_graphs <bool-variable-whether-to-draw-graphs-or-not>
```
- CNN_LSTM
```
python Models/CNN_LSTM.py --src_dir <path-to-directory-containing-preprocessed-data> --tgt_dir \
       <path-where-the-outputs-need-to-be-placed> --eq_num <equipment-number-to-be-tested> --n_test \
       <number-of-predictions-to-be-made> --n_models <numer-of-CNN_LSTM-models-to-be-made> \
       --draw_graphs <bool-variable--whether-to-draw-graphs-or-not>
```
- Ensemble
```
python Models/Ensemble.py --src_dir <path-to-directory-containing-preprocessed-data> --tgt_dir \
       <path-where-the-outputs-need-to-be-placed> --eq_num <equipment-number-to-be-tested> \
       --n_test <number-of-predictions-to-be-made> --n_models <numer-of-CNN_LSTM-models-to-be-made>\
       --draw_graphs <bool-variable--whether-to-draw-graphs-or-not>
```

### Finding Start-End Time
```
python Models/StartFinish.py --data_path <path-to-directory-containing-data-to-apply-on> --target_path <path-where-the-outputs-need-to-be-placed> --num_preds <number-of-predictions-to-be-worked-on>
```
