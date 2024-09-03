import numpy as np
np.set_printoptions(suppress=True)
import os
import sys
import json
sys.path.insert(0, "../")
sys.path.insert(0, "../../")
import pandas as pd
import datetime
from pathlib import Path
home = str(Path.home()) + "/Documents/GitHub/"

task_name = "CDDM"
RNNs_path = os.path.join(home, "trainRNNbrain", "data", "trained_RNNs", task_name)

RNNs = []
for folder in os.listdir(RNNs_path):
    if (folder == '.DS_Store'):
        pass
    else:
        if "relu" in folder:
            RNNs.append(folder)

names = []
scores = []
Ns = []
lmbdos = []
lmbdrs = []
lrs = []
activations = []
tags = []
maxiters = []
for folder in RNNs:
    day = float(datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(RNNs_path, folder))).strftime('%d'))
    month = float(datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(RNNs_path, folder))).strftime('%m'))
    year = float(datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(RNNs_path, folder))).strftime('%y'))
    if (month == 6) and (day >=22) and (year == 23):
        files = os.listdir(os.path.join(RNNs_path, folder))
        config_file = None
        for file in files:
            if "config" in file:
                config_file = file
        config_data = json.load(open(os.path.join(RNNs_path, folder, config_file), "rb+"))
        score = np.round(float(config_file.split("_")[0]), 7)
        activation = config_data["activation"]
        N = config_data["N"]
        lmbdo = config_data["lambda_orth"]
        lmbdr = config_data["lambda_r"]
        lr=config_data["lr"]
        maxiter=config_data["max_iter"]
        extra_info = f"{task_name};{activation};N={N};lmbdo={lmbdo};lmbdr={lmbdr};lr={lr};maxiter={maxiter}"
        name = f"{score}_{extra_info}"
        tag = config_data["folder_tag"]
        names.append(name)
        scores.append(score)
        Ns.append(N)
        lmbdos.append(lmbdo)
        lmbdrs.append(lmbdr)
        lrs.append(lr)
        tags.append(tag)
        activations.append(activation)
        maxiters.append(maxiter)

df = pd.DataFrame({"name" : names, "scores" : scores, "N" : Ns, "activation": activations, "lmbdo" : lmbdos, "lmbdr": lmbdrs, "lr" : lrs, "maxiter" : maxiters})
# additional filtering
df = df[df['lr'] == 0.002]
df = df[df['maxiter'] == 3000]
pd.set_option('display.max_rows', None)
df.sort_values("scores")
top_RNNs = df.sort_values("scores")["name"].tolist()

json.dump(top_RNNs, open(os.path.join(home, "trainRNNbrain", "data", f"list_of_RNNs_>={int(month)}.{int(day)}.{int(year)}.json"), 'w+'))
print(len(top_RNNs))