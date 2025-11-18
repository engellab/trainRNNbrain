from trainRNNbrain.datasaver.DataSaver import DataSaver
from trainRNNbrain.analyzers.PerformanceAnalyzer import PerformanceAnalyzer
from trainRNNbrain.trainer.Trainer_v42 import Trainer
from trainRNNbrain.rnns.RNN_numpy import RNN_numpy
from trainRNNbrain.training.training_utils import *
from trainRNNbrain.utils import jsonify, unjsonify
import time
import hydra
from matplotlib import pyplot as plt
import datetime, random
import inspect
import inspect, importlib.util, pathlib
from pathlib import Path

def import_any(target):
    p = pathlib.Path(str(target))
    if p.suffix == ".py" and p.exists():                       # file path
        spec = importlib.util.spec_from_file_location(p.stem, str(p))
        mod  = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        return mod
    # dotted path: module or module.Class
    try:
        return importlib.import_module(target)                 # module
    except Exception:
        mod_name, attr = target.rsplit(".", 1)                 # module.Class
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr)

def module_source_of(obj_or_mod):
    m = obj_or_mod if inspect.ismodule(obj_or_mod) else inspect.getmodule(obj_or_mod)
    try:
        return inspect.getsource(m)
    except OSError:
        # fallback if loaded from .pyc
        import importlib.util, pathlib
        path = pathlib.Path(getattr(m, "__file__", ""))
        if path.suffix == ".pyc":
            path = pathlib.Path(importlib.util.source_from_cache(str(path)))
        return path.read_text()

def make_tag(cfg, net_params, score, taskname):
    t = cfg.trainer
    def abbr(k):
        return 'L'+k[7:] if k.startswith('lambda_') else ''.join(w[0].upper() for w in k.split('_') if w)
    def fmt(v):
        return f"{v:.3g}" if isinstance(v, float) else str(int(v)) if isinstance(v, bool) else str(v)

    td = {k: getattr(t, k) for k in dir(t) if not k.startswith('_') and not callable(getattr(t, k))}
    lambdas = {k: v for k, v in td.items() if k.startswith('lambda_') and v is not None}
    core_keys = [k for k in ('learning_rate','lr','max_iter','dropout','drop_rate','weight_decay','orth_input_only') if k in td]

    parts = [
        f'{score}_{taskname}_{net_params["activation_name"]}',
        f'N={net_params["N"]}',
        *[f'{abbr(k)}={fmt(td[k])}' for k in core_keys],
        *[f'{abbr(k)}={fmt(lambdas[k])}' for k in sorted(lambdas)]
    ]
    return ';'.join(parts)

def filter_kwargs(callable_obj, params: dict):
    sig = inspect.signature(callable_obj)
    # if it accepts **kwargs, pass everything through
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return params
    allowed = {name for name, p in sig.parameters.items()
               if name != 'self' and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    return {k: v for k, v in params.items() if k in allowed}

OmegaConf.register_new_resolver("eval", eval)
os.environ['HYDRA_FULL_ERROR'] = '1'
@hydra.main(version_base="1.3", config_path="../../configs/", config_name=f"experimental")
def run_training(cfg: DictConfig) -> None:
    taskname = cfg.task.taskname
    seed = 'random'
    if seed == 'random':
        seed = time.time_ns() & 0xFFFFFFFF
    random.seed(int(seed))
    disp = cfg.display_figures

    # defining the task
    task_conf = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    task = hydra.utils.instantiate(task_conf)
    print(cfg)

    for i in range(cfg.n_nets):
        #defining the RNN
        rnn_config = prepare_RNN_arguments(cfg_task=cfg.task, cfg_model=cfg.model)

        rnn_config.seed = seed + (i * 14653 + (i + 65537) ** 3) % 7309
        rnn_torch = hydra.utils.instantiate(rnn_config)

        file_path = inspect.getmodule(Trainer).__file__
        suffix_trainer = file_path.split("/")[-1].split(".")[0]
        suffix_RNN = rnn_config._target_.split(".")[-2].split("_")[-1]
        tag = f"{cfg.model.activation_name}_{suffix_trainer}_{suffix_RNN}"
        print(f"training {taskname}, {tag}")
        data_save_path = os.path.join(cfg.paths.save_to, f"{taskname}_{tag}")
        os.makedirs(data_save_path, exist_ok=True)

        # defining the trainer
        optimizer = torch.optim.Adam(rnn_torch.parameters(),
                                     lr=cfg.trainer.lr,
                                     weight_decay=cfg.trainer.weight_decay)

        trainer_cfg = cfg.trainer
        items = getattr(trainer_cfg, "items", None)
        items = items() if callable(items) else vars(trainer_cfg).items()
        lambdas = {k: v for k, v in items if k.startswith("lambda_") and v is not None}

        trainer = Trainer(
            RNN=rnn_torch, Task=task,
            max_iter=trainer_cfg.max_iter, tol=trainer_cfg.tol,
            optimizer=optimizer,
            orth_input_only=trainer_cfg.orth_input_only,
            dropout=trainer_cfg.dropout,
            drop_rate=trainer_cfg.drop_rate,
            cap_s=trainer_cfg.cap_s,
            # cap_high=trainer_cfg.cap_high,
            # penalize_low=trainer_cfg.penalize_low,
            inequality_method=getattr(trainer_cfg, "inequality_method", None),
            **lambdas
        )

        mask = get_training_mask(cfg_task=cfg.task, dt=cfg.model.dt)

        # run training
        tic = time.perf_counter()
        rnn_trained, train_losses, val_losses, best_net_params, last_net_params = trainer.run_training(train_mask=mask,
                                                                                                       same_batch=cfg.trainer.same_batch)
        toc = time.perf_counter()
        print(f"Executed training in {toc - tic:0.4f} seconds")

        # At the end of training, convert everything to CPU and numpy
        for key in trainer.loss_monitor:
            # Convert each list of tensors to a single tensor
            loss_tensor = torch.tensor(trainer.loss_monitor[key])
            # Move to CPU and convert to numpy at once
            trainer.loss_monitor[key] = loss_tensor.cpu().numpy()
        for key in trainer.gradients_monitor:
            # Convert each list of tensors to a single tensor
            indicators_tensor = torch.tensor(trainer.gradients_monitor[key])
            # Move to CPU and convert to numpy at once
            trainer.gradients_monitor[key] = indicators_tensor.cpu().numpy()

        last_net_params = unjsonify(last_net_params)
        best_net_params = unjsonify(best_net_params)

        net_params = filter_kwargs(RNN_numpy, last_net_params)
        RNN_valid = RNN_numpy(**net_params, seed=seed)

        # validate
        analyzer = PerformanceAnalyzer(RNN_valid)

        def r2(x, y, axis=None, eps=1e-12):
            y_mean = np.mean(y, axis=axis, keepdims=True)
            ss_res = np.mean((x - y) ** 2, axis=axis)
            ss_tot = np.mean((y - y_mean) ** 2, axis=axis)
            return 1.0 - ss_res / (ss_tot + eps)

        input_batch_valid, target_batch_valid, conditions_valid = task.get_batch()
        print(f"torch r2 score: {trainer.eval_step(torch.from_numpy(input_batch_valid),
                                                       torch.from_numpy(target_batch_valid),
                                                       mask=mask, noise=True)}")
        score = analyzer.get_validation_score(r2,
                                              input_batch_valid,
                                              target_batch_valid,
                                              mask,
                                              sigma_rec=cfg.model.sigma_rec,
                                              sigma_inp=cfg.model.sigma_inp,
                                              seed=seed)
        score = np.round(score, 7)

        data_folder = make_tag(cfg, net_params, score, taskname)

        full_data_folder = os.path.join(data_save_path, data_folder)
        datasaver = DataSaver(full_data_folder)

        # save Trainer module
        src = module_source_of(Trainer)
        datasaver.save_data(src, "trainer.txt")
        # save RNN module from Hydra target (module or module.Class or path)
        mod_or_cls = import_any(rnn_config._target_)
        src = module_source_of(mod_or_cls)
        datasaver.save_data(src, "rnn.txt")

        print(f"r2 validation: {score}")

        if not (datasaver is None): datasaver.save_data(cfg, f"{score}_config.yaml")
        if not (datasaver is None): datasaver.save_data(jsonify(last_net_params), f"{score}_LatestParams_{taskname}.json")
        if not (datasaver is None): datasaver.save_data(jsonify(best_net_params),f"{score}_BestParams_{taskname}.json")

        fig_trainloss = plot_train_val_losses(train_losses, val_losses)
        if disp: plt.show()
        if not (datasaver is None): datasaver.save_figure(fig_trainloss, f"{score}_TrainLoss.png")

        if not (datasaver is None): datasaver.save_data(jsonify(trainer.loss_monitor), f"{score}_LossBreakdown.json")
        fig_loss_breakdown = plot_loss_breakdown(trainer.loss_monitor)
        if disp: plt.show()
        if not (datasaver is None): datasaver.save_figure(fig_loss_breakdown, f"{score}_LossBreakdown.png")

        if not (datasaver is None): datasaver.save_data(jsonify(trainer.gradients_monitor), f"{score}_GradsRaw.json")
        if not (datasaver is None): datasaver.save_data(jsonify(trainer.scaled_gradients_monitor), f"{score}_GradsScaled.json")

        fig_grads_raw = plot_loss_breakdown(trainer.gradients_monitor)
        if disp: plt.show()
        if not (datasaver is None): datasaver.save_figure(fig_grads_raw, f"{score}_GradsRaw.png")
        fig_grads_scaled = plot_loss_breakdown(trainer.scaled_gradients_monitor)
        if disp: plt.show()
        if not (datasaver is None): datasaver.save_figure(fig_grads_scaled, f"{score}_GradsScaled.png")

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

        fig_participation = analyzer.plot_participation(trajectories=trajectories)
        if disp: plt.show()
        if not (datasaver is None): datasaver.save_figure(fig_participation, "participation.png")


        dale_mask_bool = ((np.sign(np.sum(RNN_valid.W_rec, axis = 0)) + 1) / 2).astype(bool)
        dale_mask_int = (np.sign(np.sum(RNN_valid.W_rec, axis=0)) + 1)
        perm = analyzer.composite_lexicographic_sort(RNN_valid.W_inp, RNN_valid.W_rec, dale_mask_int)
        W_inp_, W_rec_, W_out_, dale_mask_bool_ = analyzer.permute_matrices(RNN_valid.W_inp,
                                                                       RNN_valid.W_rec,
                                                                       RNN_valid.W_out,
                                                                       dale_mask_bool, perm)
        analyzer.RNN.W_inp = W_inp_
        analyzer.RNN.W_rec = W_rec_
        analyzer.RNN.W_out = W_out_
        analyzer.RNN.dale_mask = dale_mask_bool_
        trajectories_, _ = analyzer.get_trajectories(input_batch_valid,
                                                     sigma_rec=0.01,
                                                     sigma_inp=0.01,
                                                     seed=seed)

        fig_matrices = analyzer.plot_matrices()
        if disp: plt.show()
        if not (datasaver is None): datasaver.save_figure(fig_matrices, "sorted_matrices.png")

        labels_ = analyzer.cluster_neurons(trajectories_, dale_mask_bool_)
        averaged_responses, grouped_dale_mask = analyzer.get_averaged_responses(trajectories_,
                                                                                dale_mask=dale_mask_bool_,
                                                                                labels=labels_)
        avg_responses = analyzer.plot_averaged_responses(averaged_responses, grouped_dale_mask)
        if disp: plt.show()
        if not (datasaver is None): datasaver.save_figure(avg_responses, "avg_responses.png")

        w_inp, w_rec, w_out = analyzer.compute_intercluster_weights(W_inp_, W_rec_, W_out_, labels_)
        fig_matrices = analyzer.plot_matrices(w_inp, w_rec, w_out)
        if disp: plt.show()
        if not (datasaver is None): datasaver.save_figure(fig_matrices, "intercluster_connectivity_matrices.png")

        ## ANIMATIONS
        # # animating trajectories:
        ani_trajectories = analyzer.animate_trajectories(trajectories)
        if disp: plt.show()
        if not (datasaver is None): datasaver.save_animation(ani_trajectories, "animated_trajectories.mp4")

        # animating selectivity:
        ani_selectivity = analyzer.animate_selectivity(trajectories=trajectories_,
                                                       axes=(0,1,2),
                                                       labels=labels_)
        if disp: plt.show()
        if not (datasaver is None): datasaver.save_animation(ani_selectivity, "animated_selectivity.mp4")



if __name__ == "__main__":
    run_training()