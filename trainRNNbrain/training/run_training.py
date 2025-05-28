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
@hydra.main(version_base="1.3", config_path="../../configs/", config_name=f"base")
def run_training(cfg: DictConfig) -> None:
    taskname = cfg.task.taskname
    tag = f"{cfg.model.activation_name}_constrained={cfg.model.constrained}"
    print(f"training {taskname}, {tag}")
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
                          max_iter=cfg.trainer.max_iter, tol=cfg.trainer.tol,
                          optimizer=optimizer, criterion=criterion,
                          lambda_orth=cfg.trainer.lambda_orth,
                          orth_input_only=cfg.trainer.orth_input_only,
                          lambda_r=cfg.trainer.lambda_r,
                          lambda_z=cfg.trainer.lambda_z)

        mask = get_training_mask(cfg_task=cfg.task, dt=cfg.model.dt)

        # run training
        tic = time.perf_counter()
        rnn_trained, train_losses, val_losses, net_params = trainer.run_training(train_mask=mask,
                                                                                 same_batch=cfg.trainer.same_batch)
        toc = time.perf_counter()
        print(f"Executed training in {toc - tic:0.4f} seconds")

        losses_history = trainer.loss_monitor
        # At the end of training, convert everything to CPU and numpy
        for key in trainer.loss_monitor:
            # Convert each list of tensors to a single tensor
            loss_tensor = torch.stack(trainer.loss_monitor[key])
            # Move to CPU and convert to numpy at once
            trainer.loss_monitor[key] = loss_tensor.cpu().numpy()

        # postprocessing and analysis
        rnn_torch, net_params = remove_silent_nodes(rnn_torch=rnn_trained,
                                                    task=task,
                                                    net_params=net_params)
        RNN_valid = RNN_numpy(**net_params)
        # validate
        analyzer = PerformanceAnalyzer(RNN_valid)
        score_function = lambda x, y: 1 - (np.mean((x - y) ** 2) / np.mean(y ** 2))
        input_batch_valid, target_batch_valid, conditions_valid = task.get_batch()
        score = analyzer.get_validation_score(score_function,
                                              input_batch_valid, target_batch_valid,
                                              mask,
                                              sigma_rec=0, sigma_inp=0)
        score = np.round(score, 7)

        data_folder = (f'{score}_{taskname}_{net_params["activation_name"]};'
                       f'N={net_params["N"]};'
                       f'lmbdO={cfg.trainer.lambda_orth};'
                       f'OrthInpOnly={cfg.trainer.orth_input_only};'
                       f'lmbdR={cfg.trainer.lambda_r};'
                       f'lmbdZ={cfg.trainer.lambda_z};'
                       f'LR={cfg.trainer.lr};'
                       f'MaxIter={cfg.trainer.max_iter}')

        full_data_folder = os.path.join(data_save_path, data_folder)
        datasaver = DataSaver(full_data_folder)
        print(f"r2 validation: {score}")

        if not (datasaver is None): datasaver.save_data(cfg, f"{score}_config.yaml")
        if not (datasaver is None): datasaver.save_data(jsonify(net_params), f"{score}_params_{taskname}.json")

        fig_trainloss = plot_train_val_losses(train_losses, val_losses)
        if disp: plt.show()
        if not (datasaver is None): datasaver.save_figure(fig_trainloss, f"{score}_train&valid_loss.png")

        if not (datasaver is None): datasaver.save_data(jsonify(trainer.loss_monitor), f"{score}_loss_breakdown.json")
        fig_loss_breakdown = plot_loss_breakdown(trainer.loss_monitor)
        if disp: plt.show()
        if not (datasaver is None): datasaver.save_figure(fig_loss_breakdown, f"{score}_loss_breakdown.png")

        inds = np.random.choice(np.arange(input_batch_valid.shape[-1]), np.minimum(input_batch_valid.shape[-1], 12))
        inputs = input_batch_valid[..., inds]
        targets = target_batch_valid[..., inds]
        conditions = [conditions_valid[ind] for ind in inds]
        fig_trials = analyzer.plot_trials(inputs, targets, mask,
                                          sigma_rec=cfg.model.sigma_rec,
                                          sigma_inp=cfg.model.sigma_inp,
                                          conditions=conditions)
        if disp: plt.show()
        if not (datasaver is None): datasaver.save_figure(fig_trials, "random_trials.png")

        trajectories, outputs = analyzer.get_trajectories(input_batch_valid)
        # # animating trajectories:
        ani_trajectories = analyzer.animate_trajectories(trajectories)
        if disp: plt.show()
        if not (datasaver is None): datasaver.save_animation(ani_trajectories, "animated_trajectories.mp4")

        if cfg.model.constrained:
            dale_mask = ((np.sign(np.sum(RNN_valid.W_rec, axis = 0)) + 1) / 2).astype(bool)
            perm = analyzer.composite_lexicographic_sort(RNN_valid.W_inp, RNN_valid.W_rec, dale_mask)
            W_inp_, W_rec_, W_out_, dale_mask_ = analyzer.permute_matrices(RNN_valid.W_inp,
                                                                           RNN_valid.W_rec,
                                                                           RNN_valid.W_out,
                                                                           dale_mask, perm)
            analyzer.RNN.W_inp = W_inp_
            analyzer.RNN.W_rec = W_rec_
            analyzer.RNN.W_out = W_out_
            analyzer.RNN.dale_mask = dale_mask_
            trajectories_, _ = analyzer.get_trajectories(input_batch_valid) # get sorted trajectories

            fig_matrices = analyzer.plot_matrices()
            if disp: plt.show()
            if not (datasaver is None): datasaver.save_figure(fig_matrices, "sorted_matrices.png")

            labels_ = analyzer.cluster_neurons(trajectories_, dale_mask_)
            averaged_responses, grouped_dale_mask = analyzer.get_averaged_responses(trajectories_,
                                                                                    dale_mask=dale_mask_,
                                                                                    labels=labels_)
            avg_responses = analyzer.plot_averaged_responses(averaged_responses, grouped_dale_mask)
            if disp: plt.show()
            if not (datasaver is None): datasaver.save_figure(avg_responses, "avg_responses.png")
        else:
            dale_mask_ = None
            labels_ = None
            trajectories_ = trajectories

        # animating selectivity:
        ani_selectivity = analyzer.animate_selectivity(trajectories=trajectories_,
                                                       axes=(0,1,2),
                                                       labels=labels_)
        if disp: plt.show()
        if not (datasaver is None): datasaver.save_animation(ani_selectivity, "animated_selectivity.mp4")


if __name__ == "__main__":
    run_training()