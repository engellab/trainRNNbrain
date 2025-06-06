from trainRNNbrain.datasaver.DataSaver import DataSaver
from trainRNNbrain.analyzers.PerformanceAnalyzer import PerformanceAnalyzer
from trainRNNbrain.trainer.Trainer import Trainer
from trainRNNbrain.rnns.RNN_numpy import RNN_numpy
from trainRNNbrain.training.training_utils import *
from trainRNNbrain.utils import jsonify
from omegaconf import DictConfig, OmegaConf
import hydra
from matplotlib import pyplot as plt
import time

OmegaConf.register_new_resolver("eval", eval)
os.environ['HYDRA_FULL_ERROR'] = '1'
taskname = "DMTS"

# Either run the script with the first decorator (from IDE), or use the second decorator to run from shell
# specifying the arguments:
# 'python run_training.py model=rnn_relu_Dale task=DMTS trainer=trainer_sequential'
# from the terminal in the folder containing this script
# @hydra.main(version_base=None, config_path="../../configs", config_name=f"training_runs/train_{taskname}")
@hydra.main(version_base="1.3", config_path="../../configs/", config_name=f"base")
def run_training(cfg: DictConfig) -> None:
    taskname = cfg.task.taskname
    tag = f"{cfg.model.activation_name}_constrained={cfg.model.constrained}"
    data_save_path = os.path.join(cfg.paths.save_to, f"{taskname}_{tag}")
    os.makedirs(data_save_path, exist_ok=True)
    disp = cfg.display_figures

    # defining the task
    task_conf = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    task = hydra.utils.instantiate(task_conf)

    for i in range(cfg.n_nets):
        #defining the RNN
        rnn_config = prepare_RNN_arguments(cfg_task=cfg.task, cfg_model=cfg.model)
        rnn_torch = hydra.utils.instantiate(rnn_config)

        # defining the trainer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(rnn_torch.parameters(),
                                     lr=cfg.trainer.lr,
                                     weight_decay=cfg.trainer.weight_decay)
        trainer = Trainer(RNN=rnn_torch, Task=task,
                          max_iter=cfg.trainer.max_iter[0], tol=cfg.trainer.tol,
                          optimizer=optimizer, criterion=criterion,
                          lambda_orth=cfg.trainer.lambda_orth,
                          orth_input_only=cfg.trainer.orth_input_only,
                          lambda_r=cfg.trainer.lambda_r[0],
                          lambda_z=cfg.trainer.lambda_z)

        mask = get_training_mask(cfg_task=cfg.task, dt=cfg.model.dt)

        # run training
        tic = time.perf_counter()
        rnn_trained, train_losses, val_losses, net_params = trainer.run_training(train_mask=mask,
                                                                                 same_batch=cfg.trainer.same_batch)
        trainer.max_iter = cfg.trainer.max_iter[1]
        trainer.lambda_r = cfg.trainer.lambda_r[1]
        trainer.RNN = rnn_trained
        rnn_trained, train_losses, val_losses, net_params = trainer.run_training(train_mask=mask,
                                                                                 same_batch=cfg.trainer.same_batch)
        toc = time.perf_counter()
        print(f"Executed training in {toc - tic:0.4f} seconds")


        # postprocessing and analysis
        rnn_torch, net_params = remove_silent_nodes(rnn_torch=rnn_trained,
                                                    task=task,
                                                    net_params=net_params)

        # validate
        RNN_valid = RNN_numpy(**net_params)
        analyzer = PerformanceAnalyzer(RNN_valid)
        score_function = lambda x, y: np.mean((x - y) ** 2)
        input_batch_valid, target_batch_valid, conditions_valid = task.get_batch()
        score = analyzer.get_validation_score(score_function, input_batch_valid, target_batch_valid, mask, sigma_rec=0,
                                              sigma_inp=0)
        score = np.round(score, 7)
        data_folder = (f'{score}_{taskname}_{net_params["activation_name"]};'
                       f'N={net_params["N"]};'
                       f'lmbdo={cfg.trainer.lambda_orth};'
                       f'orth_inp_only={cfg.trainer.orth_input_only};'
                       f'lmbdr={cfg.trainer.lambda_r[-1]};'
                       f'lr={cfg.trainer.lr};'
                       f'maxiter={cfg.trainer.max_iter[-1]}')

        full_data_folder = os.path.join(data_save_path, data_folder)
        datasaver = DataSaver(full_data_folder)

        print(f"MSE validation: {score}")

        if not (datasaver is None): datasaver.save_data(cfg, f"{score}_config.yaml")
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
        conditions = [conditions_valid[ind] for ind in inds]

        fig_trials = analyzer.plot_trials(inputs, targets, mask,
                                          sigma_rec=cfg.model.sigma_rec,
                                          sigma_inp=cfg.model.sigma_inp,
                                          conditions=conditions)
        if disp: plt.show()
        if not (datasaver is None): datasaver.save_figure(fig_trials, "random_trials.png")


if __name__ == "__main__":
    run_training()