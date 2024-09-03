from trainRNNbrain.datasaver.DataSaver import DataSaver
from trainRNNbrain.analyzers.PerformanceAnalyzer import PerformanceAnalyzer
from trainRNNbrain.trainer.Trainer import Trainer
from trainRNNbrain.rnns.RNN_numpy import RNN_numpy
from trainRNNbrain.training.training_utils import *
from trainRNNbrain.utils import jsonify
import time
import hydra
from matplotlib import pyplot as plt

OmegaConf.register_new_resolver("eval", eval)
os.environ['HYDRA_FULL_ERROR'] = '1'
taskname = "MemoryNumber"
@hydra.main(version_base="1.3", config_path="../../configs/training_runs/", config_name=f"train_{taskname}")
def run_test(cfg: DictConfig) -> None:
    taskname = cfg.task.taskname
    # defining the task
    task_conf = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    task = hydra.utils.instantiate(task_conf)
    inputs, targets, conditions = task.get_batch()

    print(inputs.shape)

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.plot(inputs[0, :, 32], color = 'r')
    ax.plot(inputs[0, :, 127], color='b')
    ax.plot(inputs[0, :, 156], color='g')
    ax.plot(inputs[0, :, 200], color='m')

    ax.plot(targets[0, :, 32], color = 'r', linestyle='--')
    ax.plot(targets[0, :, 127], color='b', linestyle='--')
    ax.plot(targets[0, :, 156], color='g', linestyle='--')
    ax.plot(targets[0, :, 200], color='m', linestyle='--')
    plt.show()



if __name__ == "__main__":
    run_test()