import colorsys
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from trainRNNbrain.analyzers.PerformanceAnalyzer import PerformanceAnalyzer
from trainRNNbrain.tasks.TaskReferenceFrame import TaskReferenceFrame
from trainRNNbrain.rnns.RNN_numpy import RNN_numpy
from matplotlib import pyplot as plt
import numpy as np
import pickle
import json
import os
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from trainRNNbrain.training.training_utils import prepare_task_arguments, get_training_mask

def condition2color(theta1):
    """
    Map two angles (theta1, theta2) onto a sphere and convert them into an RGB color.
    Both theta1 and theta2 are in the range [0, 2π].

    Parameters:
        theta1 (float): Azimuthal angle (longitude) in radians, range [0, 2π].
        theta2 (float): Polar angle (latitude) in radians, range [0, 2π].

    Returns:
        tuple: RGB color as a tuple of floats in the range [0, 1].
    """
    # Normalize theta1 to [0, 1] for hue
    hue = (theta1 % (2 * np.pi)) / (2 * np.pi)

    # Normalize theta2 to [0, 1] for saturation
    saturation = 0.8

    # Fixed value (brightness)
    value = 1.0  # Full brightness

    # Convert HSV to RGB
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)

    return (r, g, b)

# network_name = '0.9966281_ReferenceFrame_relu;N=300;lmbdo=0.25;orth_inp_only=True;lmbdr=0.05;lr=0.005;maxiter=20000'
combine_inputs = False
img_folder = "../../img/AngleAddition"
os.makedirs(img_folder, exist_ok=True)
networks_path = os.path.join('/Users/tolmach/Documents/GitHub/trainRNNbrain/data/trained_RNNs/ReferenceFrame_relu_constrained=True/ReferenceFrame_relu_constrained=True/')
coloring = "ego"

for network_name in os.listdir(networks_path):#["0.9953713_ReferenceFrame_relu;N=286;lmbdo=0.25;orth_inp_only=True;lmbdr=0.25;lr=0.005;maxiter=20000"]: #os.listdir(networks_path)
    if network_name == '.DS_Store':
        continue
    else:
        path = os.path.join(networks_path, network_name)
        score = float(network_name.split('_')[0])
        net_params = json.load(open(os.path.join(path, f"{score}_params_ReferenceFrame.json"), 'r'))
        for key in ["W_inp", "W_rec", "W_out", "y_init"]:
            net_params[key] = np.array(net_params[key])
        try:
            cfg = OmegaConf.load(open(os.path.join(path, f"{score}_config.yaml"), 'r'))
            lambda_var = cfg.trainer.lambda_var
            taskname = cfg.task.taskname

            disp = cfg.display_figures

            # defining the task
            task_conf = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
            # task_conf.batch_size = 12
            task = hydra.utils.instantiate(task_conf)
            mask = get_training_mask(cfg_task=cfg.task, dt=cfg.model.dt)

            # validate
            rnn = RNN_numpy(**net_params)
            analyzer = PerformanceAnalyzer(rnn)
            score_function = lambda x, y: np.mean((x - y) ** 2)
            input_batch, target_batch, conditions_list = task.get_batch()
            input_batch_list = [input_batch[:, :, i] for i in range(input_batch.shape[-1])]
            target_batch_list = [target_batch[:, :, i] for i in range(target_batch.shape[-1])]


            rnn.reset_state()
            rnn.clear_history()
            rnn.run(input_batch)
            trajectories = rnn.get_history()
            trajectories_flat = trajectories.reshape(trajectories.shape[0], -1)
            inputs_flat = input_batch.reshape(input_batch.shape[0], -1)
            tni_flat = np.vstack([trajectories_flat, inputs_flat])

            print(network_name, f"lambda_var = {lambda_var}")

            # pca_tuning = PCA(n_components=5)
            # pca_tuning.fit(trajectories_flat)
            # P_tuning = pca_tuning.components_.T
            # tuning = trajectories_flat @ P_tuning
            # for projection in [(0, 1, 2)]:#, (0, 1, 3), (0, 2, 3), (1, 2, 3), (2, 3, 4)]:
            #     fig = plt.figure()
            #     ax = fig.add_subplot(projection='3d')
            #     ax.scatter(*(tuning[:, p] for p in projection), color='r', s=30, edgecolor='k')
            #     plt.show()
            file = os.path.join(img_folder, f'{network_name};lambda_var={lambda_var}.mp4')
            if os.path.exists(file):
                pass
                continue
            else:
                pca_trajectories = PCA(n_components=5)
                data_to_fit = tni_flat if combine_inputs else trajectories_flat
                pca_trajectories.fit(data_to_fit.T)
                P_trajectories = pca_trajectories.components_
                data_to_reduce = np.vstack([trajectories, input_batch]) if combine_inputs else trajectories
                data_reduced = np.einsum("ji,itk->jtk", P_trajectories, data_to_reduce)[:, -1, :]

                # Create a colormap
                theta1_values = [c[f"theta_{coloring}"] for c in conditions_list]
                colors = np.array([condition2color(theta) for theta in theta1_values])

                # Initialize the figure and 3D axis
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                # Scatter plot of the point cloud
                scatter = ax.scatter(*(data_reduced[i] for i in range(3)),
                                     c=colors, marker='o', s=40, edgecolor='k')

                # Function to update the view for each frame of the animation
                def update(frame):
                    # Fast rotation: azimuth (horizontal) axis
                    azim = frame  # Full rotation every 360 frames
                    # Slow rotation: elevation (vertical) axis, 360 times slower
                    elev = frame/ 10 + frame / 60  # Starts at 10 degrees and increases very slowly
                    ax.view_init(elev=elev, azim=azim)  # Update the view
                    fig.canvas.draw()  # Force redraw the figure
                    return scatter,

                # Create the animation
                num_frames = 360 * 3
                ani = FuncAnimation(fig, update, frames=np.arange(0, num_frames, 2), interval=1, blit=False)

                # # Save the animation as a GIF file
                # ani.save(f'../../img/trajectory_endpoint_animation_{coloring}_coloring.gif', writer='pillow', fps=30, dpi=100)

                # Save the animation as an MP4 file
                ani.save(file, writer='ffmpeg', fps=30, dpi=100)
        except:
            pass







