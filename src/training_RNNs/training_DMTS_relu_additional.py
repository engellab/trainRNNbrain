import os
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
import json
from src.DataSaver import DataSaver
from src.PerformanceAnalyzer import PerformanceAnalyzer
from src.RNN_numpy import RNN_numpy
from src.utils import numpify, jsonify
from src.Trainer import Trainer
from src.RNN_torch import RNN_torch
from src.Tasks.TaskDMTS import *
from matplotlib import pyplot as plt
import torch
import time


disp = True
activation_name = "relu"
taskname = "DMTS"
train_config_file = f"train_config_{taskname}_{activation_name}.json"

from pathlib import Path
home = str(Path.home())
if home == '/home/pt1290':
    projects_folder = home
    data_save_path = home + f'/rnn_coach/data/trained_RNNs/{taskname}'
    RNN_configs_path = home + '/rnn_coach/data/configs'
elif home == '/Users/tolmach':
    projects_folder = home + '/Documents/GitHub/'
    data_save_path = projects_folder + f'/rnn_coach/data/trained_RNNs/{taskname}'
    RNN_configs_path = projects_folder + '/rnn_coach/data/configs'
else:
    pass

config_dict = json.load(
    open(os.path.join(RNN_configs_path, train_config_file), mode="r", encoding='utf-8'))

folders = ['7.25e-05_DMTS;relu;N=74;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '4.94e-05_DMTS;relu;N=94;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '7.22e-05_DMTS;relu;N=100;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001336_DMTS;relu;N=100;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '4.98e-05_DMTS;relu;N=97;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '5.59e-05_DMTS;relu;N=96;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0002235_DMTS;relu;N=99;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001634_DMTS;relu;N=83;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001376_DMTS;relu;N=97;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001001_DMTS;relu;N=98;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001992_DMTS;relu;N=93;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '8e-05_DMTS;relu;N=94;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001323_DMTS;relu;N=99;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '6.25e-05_DMTS;relu;N=100;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.000503_DMTS;relu;N=91;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '5.12e-05_DMTS;relu;N=92;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '6.53e-05_DMTS;relu;N=88;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '4.22e-05_DMTS;relu;N=97;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.000148_DMTS;relu;N=93;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '4.68e-05_DMTS;relu;N=100;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001852_DMTS;relu;N=100;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '5.1e-05_DMTS;relu;N=99;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '5.5e-05_DMTS;relu;N=89;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0003237_DMTS;relu;N=99;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '3.95e-05_DMTS;relu;N=94;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001421_DMTS;relu;N=100;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '5.16e-05_DMTS;relu;N=93;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001174_DMTS;relu;N=100;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0009019_DMTS;relu;N=92;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0002267_DMTS;relu;N=100;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '5.03e-05_DMTS;relu;N=99;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '5.43e-05_DMTS;relu;N=91;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001806_DMTS;relu;N=95;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '6.93e-05_DMTS;relu;N=99;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001539_DMTS;relu;N=78;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001311_DMTS;relu;N=98;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '7.39e-05_DMTS;relu;N=92;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '5.99e-05_DMTS;relu;N=92;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '4.71e-05_DMTS;relu;N=99;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001589_DMTS;relu;N=98;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001801_DMTS;relu;N=73;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0002795_DMTS;relu;N=98;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '9.11e-05_DMTS;relu;N=100;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '8.34e-05_DMTS;relu;N=76;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '8.83e-05_DMTS;relu;N=91;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '7.29e-05_DMTS;relu;N=100;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '7.02e-05_DMTS;relu;N=91;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '5.84e-05_DMTS;relu;N=97;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001404_DMTS;relu;N=98;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001313_DMTS;relu;N=99;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '7.92e-05_DMTS;relu;N=100;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001386_DMTS;relu;N=96;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '5.13e-05_DMTS;relu;N=99;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0004024_DMTS;relu;N=100;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '4.98e-05_DMTS;relu;N=100;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001197_DMTS;relu;N=77;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '9.21e-05_DMTS;relu;N=94;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0004109_DMTS;relu;N=94;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '5.22e-05_DMTS;relu;N=96;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '5.44e-05_DMTS;relu;N=97;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '6.36e-05_DMTS;relu;N=93;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001048_DMTS;relu;N=99;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '4.51e-05_DMTS;relu;N=90;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '5.48e-05_DMTS;relu;N=97;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '5.15e-05_DMTS;relu;N=97;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '4.93e-05_DMTS;relu;N=96;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '8.25e-05_DMTS;relu;N=85;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '6.5e-05_DMTS;relu;N=99;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001322_DMTS;relu;N=67;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '9.59e-05_DMTS;relu;N=86;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001206_DMTS;relu;N=99;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0006667_DMTS;relu;N=65;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001877_DMTS;relu;N=95;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0004128_DMTS;relu;N=79;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0003326_DMTS;relu;N=90;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001002_DMTS;relu;N=89;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001253_DMTS;relu;N=95;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001627_DMTS;relu;N=100;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001484_DMTS;relu;N=100;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '4.91e-05_DMTS;relu;N=94;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '4.95e-05_DMTS;relu;N=97;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001034_DMTS;relu;N=95;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001667_DMTS;relu;N=100;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '5.52e-05_DMTS;relu;N=91;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '4.92e-05_DMTS;relu;N=99;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '8.05e-05_DMTS;relu;N=89;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0002553_DMTS;relu;N=74;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '5.89e-05_DMTS;relu;N=99;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '8.01e-05_DMTS;relu;N=98;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '6.25e-05_DMTS;relu;N=92;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '8.59e-05_DMTS;relu;N=99;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0002028_DMTS;relu;N=97;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '5.29e-05_DMTS;relu;N=97;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '4.28e-05_DMTS;relu;N=99;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001285_DMTS;relu;N=93;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '6.12e-05_DMTS;relu;N=82;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '9.14e-05_DMTS;relu;N=96;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0004667_DMTS;relu;N=97;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '4.6e-05_DMTS;relu;N=95;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001298_DMTS;relu;N=99;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001876_DMTS;relu;N=90;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '4.74e-05_DMTS;relu;N=97;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '5.19e-05_DMTS;relu;N=99;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001408_DMTS;relu;N=66;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.000165_DMTS;relu;N=95;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '9.13e-05_DMTS;relu;N=92;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '8.21e-05_DMTS;relu;N=95;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '4.47e-05_DMTS;relu;N=99;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '6.04e-05_DMTS;relu;N=96;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '4.79e-05_DMTS;relu;N=97;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001701_DMTS;relu;N=91;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '4.73e-05_DMTS;relu;N=88;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '8.4e-05_DMTS;relu;N=100;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '5.81e-05_DMTS;relu;N=96;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '6.07e-05_DMTS;relu;N=86;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '4.26e-05_DMTS;relu;N=93;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '5.11e-05_DMTS;relu;N=91;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001935_DMTS;relu;N=98;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '4.39e-05_DMTS;relu;N=100;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '5.17e-05_DMTS;relu;N=100;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '6.47e-05_DMTS;relu;N=87;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0002088_DMTS;relu;N=96;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '5.47e-05_DMTS;relu;N=95;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001317_DMTS;relu;N=93;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0005946_DMTS;relu;N=83;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '5.29e-05_DMTS;relu;N=91;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '9.89e-05_DMTS;relu;N=99;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '5e-05_DMTS;relu;N=89;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '5.33e-05_DMTS;relu;N=95;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '8e-05_DMTS;relu;N=96;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '4.42e-05_DMTS;relu;N=94;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '6.45e-05_DMTS;relu;N=97;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0006498_DMTS;relu;N=89;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '7.2e-05_DMTS;relu;N=92;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000',
           '0.0001249_DMTS;relu;N=94;lmbdo=0.3;orth_inp_only=True;lmbdr=0.0;lr=0.01;maxiter=5000']
try:
    job_id = int(os.environ["SLURM_ARRAY_JOB_ID"])
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    print(job_id, task_id)
except:
    task_id = 1

for folder in folders[(task_id - 1) * 10: np.minimum(len(folders), task_id * 10)]:
    if activation_name in folder:
        files = os.listdir(os.path.join(data_save_path, folder))
        for file in files:
            if "params" in file:
                net_params = json.load(open(os.path.join(data_save_path, folder, file), "rb+"))
                score_old = file.split("_")[0]
        # defining RNN:
        activation_name = config_dict["activation"]
        match activation_name:
            case 'relu': activation = lambda x: torch.maximum(torch.tensor(0.0), x)
            case 'tanh': activation = lambda x: torch.tanh(x)
            case 'sigmoid': activation = lambda x: 1 / (1 + torch.exp(-x))
            case 'softplus': activation = lambda x: torch.log(1 + torch.exp(5 * x))

        dt = config_dict["dt"]
        tau = config_dict["tau"]
        constrained = config_dict["constrained"]
        connectivity_density_rec = config_dict["connectivity_density_rec"]
        spectral_rad = config_dict["sr"]
        sigma_inp = config_dict["sigma_inp"]
        sigma_rec = config_dict["sigma_rec"]
        # seed = config_dict["seed"]
        seed = None
        N = net_params["N"]

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        rng = torch.Generator(device=torch.device(device))

        if not seed is None:
            rng.manual_seed(seed)
        else:
            rng.manual_seed(np.random.randint(100000))

        input_size = config_dict["num_inputs"]
        output_size = config_dict["num_outputs"]

        # Task:
        n_steps = config_dict["n_steps"]
        task_params = config_dict["task_params"]

        # Trainer:
        lambda_orth = config_dict["lambda_orth"]
        orth_input_only = config_dict["orth_input_only"]
        lambda_r = config_dict["lambda_r"]
        mask = np.array(config_dict["mask"])
        max_iter = config_dict["max_iter"]
        tol = config_dict["tol"]
        lr = config_dict["lr"]
        weight_decay = config_dict["weight_decay"]
        same_batch = config_dict["same_batch"]

        # General:
        folder_tag = config_dict["folder_tag"]

        # # creating instances:
        rnn_torch = RNN_torch(N=N, dt=dt, tau=tau, input_size=input_size, output_size=output_size,
                              activation=activation, constrained=constrained,
                              sigma_inp=sigma_inp, sigma_rec=sigma_rec,
                              connectivity_density_rec=connectivity_density_rec,
                              spectral_rad=spectral_rad,
                              random_generator=rng)
        rnn_torch.W_inp.data = torch.from_numpy(np.array(net_params["W_inp"]).astype("float32")).to(device)
        rnn_torch.W_rec.data = torch.from_numpy(np.array(net_params["W_rec"]).astype("float32")).to(device)
        rnn_torch.W_out.data = torch.from_numpy(np.array(net_params["W_out"]).astype("float32")).to(device)
        task = TaskDMTS(n_steps=n_steps, n_inputs=input_size, n_outputs=output_size, task_params=task_params)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(rnn_torch.parameters(),
                                     lr=lr,
                                     weight_decay=weight_decay)
        trainer = Trainer(RNN=rnn_torch, Task=task,
                          max_iter=max_iter, tol=tol,
                          optimizer=optimizer, criterion=criterion,
                          lambda_orth=lambda_orth, orth_input_only=orth_input_only,
                          lambda_r=lambda_r)

        tic = time.perf_counter()
        rnn_trained, train_losses, val_losses, net_params = trainer.run_training(train_mask=mask, same_batch=same_batch)
        toc = time.perf_counter()
        print(f"Executed training in {toc - tic:0.4f} seconds")

        # throw out all the silent neurons!
        ######### clean the RNN from silent neurons!
        input_batch, target_batch, conditions = task.get_batch()
        rnn_torch.sigma_rec = rnn_torch.sigma_inp = torch.tensor(0, device=rnn_torch.device)
        y, predicted_output_rnn = rnn_torch(torch.from_numpy(input_batch.astype("float32")).to(rnn_torch.device))
        Y = torch.hstack([y.detach()[:, :, i] for i in range(y.shape[-1])]).T
        Y_mean = torch.mean(torch.abs(Y), axis=0)
        inds_fr = (torch.where(Y_mean > 0)[0]).tolist()
        N_reduced = len(inds_fr)
        config_dict["N"] = N_reduced
        N = N_reduced
        W_rec = net_params["W_rec"][inds_fr, :]
        W_rec = W_rec[:, inds_fr]
        net_params["W_rec"] = deepcopy(W_rec)
        W_out = net_params["W_out"][:, inds_fr]
        net_params["W_out"] = deepcopy(W_out)
        W_inp = net_params["W_inp"][inds_fr, :]
        net_params["W_inp"] = deepcopy(W_inp)
        net_params["bias_rec"] = None
        net_params["y_init"] = np.zeros(N_reduced)
        RNN_params = {"W_inp": np.array(net_params["W_inp"]),
                      "W_rec": np.array(net_params["W_rec"]),
                      "W_out": np.array(net_params["W_out"]),
                      "b_rec": np.array(net_params["bias_rec"]),
                      "y_init": np.zeros(N)}
        net_params["N"] = N_reduced
        rnn_trained.set_params(RNN_params)
        ########

        # validate
        RNN_valid = RNN_numpy(N=net_params["N"],
                              dt=net_params["dt"],
                              tau=net_params["tau"],
                              activation=numpify(activation),
                              W_inp=net_params["W_inp"],
                              W_rec=net_params["W_rec"],
                              W_out=net_params["W_out"],
                              bias_rec=net_params["bias_rec"],
                              y_init=net_params["y_init"])

        analyzer = PerformanceAnalyzer(RNN_valid)
        score_function = lambda x, y: np.mean((x - y) ** 2)
        input_batch_valid, target_batch_valid, conditions_valid = task.get_batch()
        score = analyzer.get_validation_score(score_function, input_batch_valid, target_batch_valid, mask, sigma_rec=0, sigma_inp=0)
        score = np.round(score, 7)

        data_folder = f'{score}_{taskname};{activation_name};N={N_reduced};lmbdo={lambda_orth};orth_inp_only={orth_input_only};lmbdr={lambda_r};lr={lr};maxiter={max_iter}'
        # data_folder = os.path.join(data_save_path, folder)
        if folder_tag != '':
            data_folder+=f";tag={folder_tag}"
        full_data_folder = os.path.join(data_save_path, data_folder)
        datasaver = DataSaver(full_data_folder)


        print(f"MSE validation: {score}")
        if not (datasaver is None): datasaver.save_data(jsonify(config_dict), f"{score}_config.json")
        if not (datasaver is None): datasaver.save_data(jsonify(net_params), f"{score}_params_{taskname}.json")

        fig_trainloss = plt.figure(figsize=(10, 3))
        plt.plot(train_losses, color='r', label='train loss (log scale)')
        plt.plot(val_losses, color='b', label='valid loss (log scale)')
        plt.yscale("log")
        plt.grid(True)
        plt.legend(fontsize=16)
        if disp: plt.show()
        if not (datasaver is None): datasaver.save_figure(fig_trainloss, f"{score}_train&valid_loss.png")

        batch_size = input_batch_valid.shape[2]
        RNN_valid.clear_history()
        RNN_valid.run(input_timeseries=input_batch_valid, sigma_rec=0, sigma_inp=0)
        RNN_trajectories = RNN_valid.get_history()
        RNN_output = RNN_valid.get_output()
        trajecory_data = {}
        trajecory_data["inputs"] = input_batch_valid
        trajecory_data["trajectories"] = RNN_trajectories
        trajecory_data["outputs"] = RNN_output
        trajecory_data["targets"] = target_batch_valid
        trajecory_data["conditions"] = conditions_valid
        datasaver.save_data(trajecory_data, f"{score}_RNNtrajdata_{taskname}.pkl")

        print(f"Plotting random trials")
        inds = np.random.choice(np.arange(input_batch_valid.shape[-1]), 12)
        inputs = input_batch_valid[..., inds]
        targets = target_batch_valid[..., inds]

        fig_trials = analyzer.plot_trials(inputs, targets, mask, sigma_rec=sigma_rec, sigma_inp=sigma_inp)
        if disp:
            plt.show()
        if not (datasaver is None): datasaver.save_figure(fig_trials, "random_trials.png")

        # dsa = DynamicSystemAnalyzer(RNN_valid)
        # params = {"fun_tol": 0.05,
        #           "diff_cutoff": 1e-4,
        #           "sigma_init_guess": 5,
        #           "patience": 50,
        #           "stop_length": 50,
        #           "mode": "approx"}
        # dsa.get_fixed_points(Input=np.zeros(input_size), **params)
        # fig_fp = dsa.plot_fixed_points(projection='2D')
        # if disp:
        #     plt.show()
        # if not (datasaver is None): datasaver.save_figure(fig_fp, "fp_projection")
