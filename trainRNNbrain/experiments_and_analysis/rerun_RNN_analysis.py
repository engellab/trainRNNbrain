from trainRNNbrain.datasaver.DataSaver import DataSaver
from trainRNNbrain.analyzers.PerformanceAnalyzer import PerformanceAnalyzer
from trainRNNbrain.trainer.Trainer_v3 import Trainer
from trainRNNbrain.rnns.RNN_numpy import RNN_numpy
from trainRNNbrain.training.training_utils import *
from trainRNNbrain.utils import jsonify, unjsonify
import time
import hydra
from matplotlib import pyplot as plt

OmegaConf.register_new_resolver("eval", eval)
os.environ['HYDRA_FULL_ERROR'] = '1'
@hydra.main(version_base="1.3", config_path="../../configs/", config_name=f"base")
def rerun_analysis(cfg: DictConfig) -> None:
    RNN_list = ["0.9020111_CDDM_relu;N=500;lmbdO=0.3;OrthInpOnly=True;lmbdR=0.5;lmbdZ=1.0;LR=0.005;MaxIter=60000",
                "0.9169394_CDDM_relu;N=500;lmbdO=0.3;OrthInpOnly=True;lmbdR=0.5;lmbdZ=1.0;LR=0.005;MaxIter=60000",
                "0.9522767_CDDM_relu;N=500;lmbdO=0.3;OrthInpOnly=True;lmbdR=0.5;lmbdZ=1.0;LR=0.005;MaxIter=60000",
                "0.9569778_CDDM_relu;N=500;lmbdO=0.3;OrthInpOnly=True;lmbdR=0.5;lmbdZ=1.0;LR=0.005;MaxIter=60000",
                "0.9648264_CDDM_relu;N=500;lmbdO=0.3;OrthInpOnly=True;lmbdR=0.5;lmbdZ=1.0;LR=0.005;MaxIter=60000",
                "0.9756123_CDDM_relu;N=500;lmbdO=0.3;OrthInpOnly=True;lmbdR=0.5;lmbdZ=1.0;LR=0.005;MaxIter=60000",
                "0.9779438_CDDM_relu;N=500;lmbdO=0.3;OrthInpOnly=True;lmbdR=0.5;lmbdZ=1.0;LR=0.005;MaxIter=60000",
                "0.9793417_CDDM_relu;N=500;lmbdO=0.3;OrthInpOnly=True;lmbdR=0.5;lmbdZ=1.0;LR=0.005;MaxIter=60000",
                "0.9840524_CDDM_relu;N=500;lmbdO=0.3;OrthInpOnly=True;lmbdR=0.5;lmbdZ=1.0;LR=0.005;MaxIter=60000",
                "0.9852374_CDDM_relu;N=500;lmbdO=0.3;OrthInpOnly=True;lmbdR=0.5;lmbdZ=1.0;LR=0.005;MaxIter=60000",
                "0.9855113_CDDM_relu;N=500;lmbdO=0.3;OrthInpOnly=True;lmbdR=0.5;lmbdZ=1.0;LR=0.005;MaxIter=60000",
                "0.9859854_CDDM_relu;N=500;lmbdO=0.3;OrthInpOnly=True;lmbdR=0.5;lmbdZ=1.0;LR=0.005;MaxIter=60000",
                "0.986649_CDDM_relu;N=500;lmbdO=0.3;OrthInpOnly=True;lmbdR=0.5;lmbdZ=1.0;LR=0.005;MaxIter=60000",
                "0.9871337_CDDM_relu;N=500;lmbdO=0.3;OrthInpOnly=True;lmbdR=0.5;lmbdZ=1.0;LR=0.005;MaxIter=60000",
                "0.9874427_CDDM_relu;N=500;lmbdO=0.3;OrthInpOnly=True;lmbdR=0.5;lmbdZ=1.0;LR=0.005;MaxIter=60000",
                "0.9877549_CDDM_relu;N=500;lmbdO=0.3;OrthInpOnly=True;lmbdR=0.5;lmbdZ=1.0;LR=0.005;MaxIter=60000",
                "0.9884098_CDDM_relu;N=500;lmbdO=0.3;OrthInpOnly=True;lmbdR=0.5;lmbdZ=1.0;LR=0.005;MaxIter=60000",
                "0.9901413_CDDM_relu;N=500;lmbdO=0.3;OrthInpOnly=True;lmbdR=0.5;lmbdZ=1.0;LR=0.005;MaxIter=60000",
                "0.9907682_CDDM_relu;N=500;lmbdO=0.3;OrthInpOnly=True;lmbdR=0.5;lmbdZ=1.0;LR=0.005;MaxIter=60000",
                "0.9910536_CDDM_relu;N=500;lmbdO=0.3;OrthInpOnly=True;lmbdR=0.5;lmbdZ=1.0;LR=0.005;MaxIter=60000",
                "0.991495_CDDM_relu;N=500;lmbdO=0.3;OrthInpOnly=True;lmbdR=0.5;lmbdZ=1.0;LR=0.005;MaxIter=60000"]


    taskname = cfg.task.taskname
    tag = f"{cfg.model.activation_name}_constrained={cfg.model.constrained}"
    print(f"training {taskname}, {tag}")
    # data_save_path = os.path.join(cfg.paths.save_to, f"{taskname}_{tag}")
    data_save_path = '/Users/tolmach/Documents/GitHub/trainRNNbrain/data/trained_RNNs/CDDM_relu_constrained=True_v3'
    disp = cfg.display_figures

    # defining the task
    task_conf = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    task = hydra.utils.instantiate(task_conf)

    for i, data_folder in enumerate(RNN_list):
        score = data_folder.split("_")[0]
        net_params_file = f"{score}_params_CDDM.json"
        with open(os.path.join(data_save_path, data_folder, net_params_file), 'r') as f:
            net_params = unjsonify(json.load(f))

        mask = get_training_mask(cfg_task=cfg.task, dt=cfg.model.dt)
        RNN_valid = RNN_numpy(**net_params)

        analyzer = PerformanceAnalyzer(RNN_valid)
        score_function = lambda x, y: 1 - (np.mean((x - y) ** 2) / np.mean(y ** 2))
        input_batch_valid, target_batch_valid, conditions_valid = task.get_batch()
        score = analyzer.get_validation_score(score_function,
                                              input_batch_valid, target_batch_valid,
                                              mask,
                                              sigma_rec=0, sigma_inp=0)
        score = np.round(score, 7)
        full_data_folder = os.path.join(data_save_path, data_folder)
        datasaver = DataSaver(full_data_folder)
        print(f"r2 validation: {score}")

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

        if cfg.model.constrained:
            dale_mask = ((np.sign(np.sum(RNN_valid.W_rec, axis = 0)) + 1) / 2).astype(bool)
            perm = analyzer.composite_lexicographic_sort(RNN_valid.W_inp, RNN_valid.W_rec, dale_mask.astype(int))
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

            labels_ = analyzer.cluster_neurons(trajectories_, dale_mask_, n_clusters=(12, 6))
            averaged_responses, grouped_dale_mask = analyzer.get_averaged_responses(trajectories_,
                                                                                    dale_mask=dale_mask_.astype(int),
                                                                                    labels=labels_)
            avg_responses = analyzer.plot_averaged_responses(averaged_responses, grouped_dale_mask)
            if disp: plt.show()
            if not (datasaver is None): datasaver.save_figure(avg_responses, "avg_responses.png")

            w_inp, w_rec, w_out = analyzer.compute_intercluster_weights(W_inp_, W_rec_, W_out_, labels_)
            fig_matrices = analyzer.plot_matrices(w_inp, w_rec, w_out)
            if disp: plt.show()
            if not (datasaver is None): datasaver.save_figure(fig_matrices, "intercluster_connectivity_matrices.png")
        else:
            dale_mask_ = None
            labels_ = None
            # trajectories_ = trajectories


        ### ANIMATIONS
        # trajectories, outputs = analyzer.get_trajectories(input_batch_valid)
        # # # animating trajectories:
        # ani_trajectories = analyzer.animate_trajectories(trajectories)
        # if disp: plt.show()
        # if not (datasaver is None): datasaver.save_animation(ani_trajectories, "animated_trajectories.mp4")
        #
        # # animating selectivity:
        # ani_selectivity = analyzer.animate_selectivity(trajectories=trajectories_,
        #                                                axes=(0,1,2),
        #                                                labels=labels_)
        # if disp: plt.show()
        # if not (datasaver is None): datasaver.save_animation(ani_selectivity, "animated_selectivity.mp4")


if __name__ == "__main__":
    rerun_analysis()