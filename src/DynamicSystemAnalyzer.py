from copy import deepcopy
import numpy as np
import scipy.optimize
from matplotlib import pyplot as plt
from scipy.optimize import fsolve, minimize
from tqdm.auto import tqdm
from src.utils import get_colormaps, in_the_list, sort_eigs, make_orientation_consistent
import warnings
warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA

class DynamicSystemAnalyzer():
    '''
    Generic class for analysis of the RNN dynamics: finding fixed points and plotting them in 2D and 3D
    '''
    def __init__(self, RNN_numpy):
        self.RNN = RNN_numpy
        self.rhs = self.RNN.rhs_noisless
        self.rhs_jac = self.RNN.rhs_jac
        self.fp_data = {}

        def objective(x, Input):
            return np.sum(self.rhs(x, Input) ** 2)

        def objective_grad(x, Input):
            return 2 * (self.rhs(x, Input).reshape(1, -1) @ self.rhs_jac(x, Input)).flatten()

        self.objective = objective
        self.objective_grad = objective_grad

    def get_fixed_points(self, Input,
                         patience=100,
                         fun_tol=1e-12,
                         stop_length=100,
                         sigma_init_guess=10,
                         eig_cutoff=1e-10,
                         diff_cutoff=1e-7,
                         mode='exact'):
        '''
        calculates fixed points (stable, unstable and marginally stable)
        :param Input: np.array, input to the RNN at which the fixed points are calculated
        :param patience: the greater the patience parameter, the longer the fixed point are searched for. The search for
        the fixed points (FPs) terminates if patience is exceeded
        :param fun_tol: RHS norm tolerance parameter: the points with the values greate than fun_tol are discarded
        :param stop_length: the maximal number of fixed points. The search terminates if number of found FPs
        exceeds the stop_length parameter
        :param sigma_init_guess: the variance of the N-dimensional Gausssian distribution for generating an initial guess
        for a FP
        :param eig_cutoff: if the norm of the max eigenvalue of the Jacobian at a found FP is lesser than eig_cutoff,
        count this FP as "marginally stable"
        :param diff_cutoff: if the difference of the currently found FP with any of the previously found FPs
        (stored in the list) is lesser than diff_cutoff - discard this point as a duplicate
        :return: arrays of points: stable fixed points, unstable fixed points, marginally stable fixed points.
        with dimensions (num_points x N) in each array
        :param mode: 'exact' or 'approx'. 'exact' - computes exact fixed points, with scipy.optimize.fsolve method
        'approx' finds 'slow points' - points with small |RHS|^2, with fun_tol controlling the cut-off |RHS|^2.
        '''

        unstable_fps = []; stable_fps = []; marginally_stable_fps = []; all_points = []
        N = self.RNN.W_rec.shape[0]
        cntr = 0 # counter parameter, keeps track of how many times in a row an optimizer didn't find any new fp
        # proceed while (cntr <= patience) and unless one of the list start to overflow (because of a 2d attractor)
        while (cntr <= patience) and (len(all_points) < stop_length):
            x0 = sigma_init_guess * np.random.randn(N)
            # finding the roots of RHS of the RNN
            if mode == 'exact':
                x_root = fsolve(func=self.rhs, x0=x0, fprime=self.rhs_jac, args=(Input,))
            elif mode == "approx":
                res = scipy.optimize.minimize(fun=self.objective, x0=x0, args=(Input, ), jac=self.objective_grad, method='Powell')
                x_root = res.x
            else:
                raise ValueError(f"Mode {mode} is not implemented!")
            fun_val = self.objective(x_root, Input)
            cntr += 1
            if fun_val <= fun_tol:
                J = self.rhs_jac(x_root, Input)
                L = np.linalg.eigvals(J)
                L_0 = np.max(np.real(L))
                if not in_the_list(x_root, all_points, diff_cutoff=diff_cutoff):
                    cntr = 0
                    all_points.append(x_root)
                    if (np.abs(L_0) <= eig_cutoff): # marginally stable fixed point (belongs to 1D attractor)
                        marginally_stable_fps.append(x_root)
                    else:
                        stable_fps.append(x_root) if (L_0 < -eig_cutoff) else unstable_fps.append(x_root)
        # Saving the data in the internal dictionary accessible by the input vector turned into string:
        input_as_key = str(Input.tolist())
        self.fp_data[input_as_key] = {}
        self.fp_data[input_as_key]["stable_fps"] = np.array(stable_fps)
        self.fp_data[input_as_key]["unstable_fps"] = np.array(unstable_fps)
        self.fp_data[input_as_key]["marginally_stable_fps"] = np.array(marginally_stable_fps)
        return np.array(unstable_fps), np.array(stable_fps), np.array(marginally_stable_fps)

    def plot_fixed_points(self, Input,
                          patience=100,
                          fun_tol=1e-12,
                          stop_length=20,
                          sigma_init_guess=10,
                          eig_cutoff=1e-10,
                          diff_cutoff=1e-7,
                          projection='2D'):
        '''
        a function that calculated the fixed points if they are not yet calculated.
        Performs PCA on them and then plots these points projected on the first PC

        same parameters as in 'get_fixed_points' function
        :param projection: to plot the FPs either on a plane (2D) or in 3D
        :return: a figure of fixed points on the first
        '''
        n_dim = 2 if projection == '2D' else 3
        input_as_key = str(Input.tolist())
        # if the fixed points are not yet calculated:
        if (len(list(self.fp_data[input_as_key].keys())) == 0):
            self.get_fixed_points(Input=Input,
                                  patience=patience,
                                  fun_tol=fun_tol,
                                  stop_length=stop_length,
                                  sigma_init_guess=sigma_init_guess,
                                  eig_cutoff=eig_cutoff,
                                  diff_cutoff=diff_cutoff)
        point_type_keys = ["stable_fps", "unstable_fps", "marginally_stable_fps"]
        points_quantity = [self.fp_data[input_as_key][key].shape[0] for key in point_type_keys]
        types_with_nonzero_points = []
        for i in range(len(points_quantity)):
            if points_quantity[i] != 0:
                types_with_nonzero_points.append(point_type_keys[i])
        if (len(types_with_nonzero_points) == 0):
            raise ValueError("Didn't find any fixed points!")
        points = np.vstack([self.fp_data[input_as_key][key].reshape(-1, self.RNN.N) for key in types_with_nonzero_points])
        if points.shape[0] < n_dim:
            raise ValueError("The number of found fixed points is lesser than n_dim of projection!")
        pca = PCA(n_components=n_dim)
        pca.fit(points)
        P = np.array(pca.components_)
        # projecting fixed points onto n_dim-subspace
        data_to_plot = {}
        for key in types_with_nonzero_points:
            data_to_plot[key] = self.fp_data[input_as_key][key] @ P.T

        # Plotting the fixed points
        if n_dim == 2:
            fig = plt.figure(figsize=(7, 7))
            fig.suptitle(r"Fixed points projected on 2D PCA plane", fontsize=16)
            markers = ["o", "x", "o"]; colors = ["blue", "red", "k"]
            for k, key in enumerate(types_with_nonzero_points):
                if self.fp_data[input_as_key][key].shape[0] != 0:
                    for i in range(data_to_plot[key].shape[0]):
                        plt.scatter(data_to_plot[key][i, 0],
                                    data_to_plot[key][i, 1],
                                    marker=markers[k], s=100, color=colors[k], edgecolors='k')
            plt.ylabel("PC 1", fontsize=16)
            plt.xlabel("PC 2", fontsize=16)
            plt.grid(True)
        elif n_dim == 3:
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(projection='3d')
            # make the panes transparent
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.set_xlabel("PC 1", fontsize=20)
            ax.set_ylabel("PC 2", fontsize=20)
            ax.set_zlabel("PC 3", fontsize=20)
            fig.suptitle(r"Fixed points projected on 3D PCA subspace", fontsize=16)
            markers = ["o", "x", "o"]; colors = ["blue", "red", "k"]
            for k, key in enumerate(types_with_nonzero_points):
                if self.fp_data[input_as_key][key].shape[0] != 0:
                    for i in range(data_to_plot[key].shape[0]):
                        ax.scatter(data_to_plot[key][i, 0],
                                    data_to_plot[key][i, 1],
                                    data_to_plot[key][i, 2],
                                    marker=markers[k], s=100, color=colors[k], edgecolors='k')
        return fig

    def compute_point_analytics(self, point, Input):
        '''
        :param point: a numpy vector, specifying a point of the state-space of RNN
        :param Input: a numpy vector, specifying the input to the RNN
        :return: fun_val - value of the |RHS|^2, J - Jacobian, E - eigenvalues sorted according their real part,
        the first left eigenvector l and the first right eigenvector r
        '''
        self.RNN.y = deepcopy(point)
        fun_val = self.objective(point, Input)
        J = self.rhs_jac(point, Input)
        E, R = np.linalg.eig(J)
        E, R = sort_eigs(E, R)
        L = np.linalg.inv(R)
        l = np.real(L[0, :])
        r = np.real(R[:, 0])
        return fun_val, J, E, l, r

class DynamicSystemAnalyzerCDDM(DynamicSystemAnalyzer):
    '''
    Class which is inрerited from the DynamicSystemAnalyzer base class,
    dedicated to processing of the RNNs trained on CDDM task
    '''
    def __init__(self, RNN):
        DynamicSystemAnalyzer.__init__(self, RNN)
        self.choice_axis = self.RNN.W_out.flatten() if self.RNN.W_out.shape[0] == 1 \
            else (self.RNN.W_out[0, :] - self.RNN.W_out[1, :])
        self.context_axis = self.RNN.W_inp[:, 0] - self.RNN.W_inp[:, 1]
        self.sensory_axis = np.sum([self.RNN.W_inp[:, i] for i in [2, 3, 4, 5]])

        self.data = {}
        self.data["motion"] = {}
        self.data["color"] = {}

    def calc_fixed_points_CDDM(self, patience=10,
                              fun_tol=1e-12,
                              stop_length=100,
                              sigma_init_guess=10,
                              eig_cutoff=1e-10,
                              diff_cutoff=1e-7):
        '''
        Get fixed points for two different contexts: "motion" and "color",
        each corresponding to a different input to the RNN
        '''
        default_input = np.array([0, 0, 0.5, 0.5, 0.5, 0.5])
        for context in ["motion", "color"]:
            ctxt_ind = 0 if context == 'motion' else 1
            # generate context-relevant input
            Input = deepcopy(default_input)
            Input[ctxt_ind] = 1
            # find fixed points for each input
            unstable_fps, stable_fps, marginally_stable_fps = self.get_fixed_points(Input,
                                                                                    patience=patience,
                                                                                    fun_tol=fun_tol,
                                                                                    stop_length=stop_length,
                                                                                    sigma_init_guess=sigma_init_guess,
                                                                                    eig_cutoff=eig_cutoff,
                                                                                    diff_cutoff=diff_cutoff)
            self.data[context]["unstable_fps"] = deepcopy(unstable_fps)
            self.data[context]["stable_fps"] = deepcopy(stable_fps)
            self.data[context]["marginally_stable_fps"] = deepcopy(marginally_stable_fps)
        return None

    def get_LineAttractor_endpoints(self, context, nudge=0.05, T_steps=1000, relax_steps=10):
        '''
        :param context: "motion" or "color"
        :param nudge: an additional input, creating a bias either to the right choice, or to the left choice
        :param T_steps: run the RNN with the applied input for the T_steps duration
        :param relax_steps: after T_steps with input, turn off any input and let the RNN relax to 'slow points'
        :return: left-most and right-most points of a presupposed line attractor
        '''
        ctxt_ind = 0 if context == 'motion' else 1
        sensory_inds = [2, 3] if context == 'motion' else [4, 5]
        default_input = np.array([0, 0, 0.5, 0.5, 0.5, 0.5])
        default_input[ctxt_ind] = 1.0
        input_right_decision = deepcopy(default_input)
        input_left_decision = deepcopy(default_input)
        input_right_decision[sensory_inds] += np.array([nudge, -nudge])
        input_left_decision[sensory_inds] -= np.array([nudge, -nudge])
        points = []
        # find left and right points
        for inp in [input_left_decision, input_right_decision]:
            self.RNN.y_init = deepcopy(self.RNN.y_init)
            self.RNN.y = deepcopy(self.RNN.y_init)
            self.RNN.run(input_timeseries=np.repeat(inp[:, np.newaxis], axis=1, repeats=T_steps))
            self.RNN.run(input_timeseries=np.repeat(inp[:, np.newaxis], axis=1, repeats=relax_steps))
            points.append(deepcopy(self.RNN.y))
        return points[0], points[1]

    def calc_LineAttractor_analytics(self,
                                     N_points=31,
                                     patience=10,
                                     fun_tol=1e-12,
                                     stop_length=20,
                                     sigma_init_guess=10,
                                     eig_cutoff=1e-10,
                                     diff_cutoff=1e-7,
                                     obj_max_iter=100,
                                     nudge=0.05,
                                     T_steps=1000,
                                     relax_steps=10):
        '''
        :param N_points: number of points on each line attractor
        :param obj_max_iter:  maximum iteration in the |RHS|^2 minimization process
        the rest of the parameters are the same as in 'get_LineAttractor_endpoints', 'get_fixed_points'
        :return: a dictionary with "color" and "motion" contexts, each containing sub-dictionary with:
        'slow points', |RHS|^2 value, Jacobian, eigenvalues, the principal left and right eigenvectors over these points
        '''
        default_input = np.array([0, 0, 0.5, 0.5, 0.5, 0.5])
        for context in ["motion", "color"]:
            ctxt_ind = 0 if context == 'motion' else 1
            Input = deepcopy(default_input)
            Input[ctxt_ind] = 1
            # if the fixed points are not yet calculated
            if (len(list(self.data[context].keys())) == 0):
                self.calc_fixed_points_CDDM(patience=patience,
                                       fun_tol=fun_tol,
                                       stop_length=stop_length,
                                       sigma_init_guess=sigma_init_guess,
                                       eig_cutoff=eig_cutoff,
                                       diff_cutoff=diff_cutoff)
            #get the end points of the line attractor
            left_point, right_point = self.get_LineAttractor_endpoints(context,
                                                                       nudge=nudge,
                                                                       T_steps=T_steps,
                                                                       relax_steps=relax_steps)
            # define the starting direction for search
            increment = (1 / (N_points - 1)) * (right_point - left_point)
            direction = deepcopy(increment)
            direction *= np.sign(np.dot(self.choice_axis, direction))
            slow_points = []; fun_vals = []; eigs = []; jacs = []; selection_vects = []; principle_eigenvects = []
            x_init = deepcopy(left_point)
            print(f"Analyzing points on a line attractor in {context} context...")
            for i in tqdm(range(N_points)):
                # minimize ||RHS(x)|| such that the x stays within a space orthogonal to the line attractor
                res = minimize(self.objective, x0=x_init, args=(Input,), method='SLSQP',
                               jac=self.objective_grad, options={'disp': False, 'maxiter': obj_max_iter},
                               constraints={'type': 'eq', 'fun': lambda x: np.dot(x - x_init, increment)})
                x_root = deepcopy(res.x)
                x_init = x_root + increment
                slow_pt = deepcopy(x_root)

                # compute analytics at the slow point:
                fun_val, J, E, l, r = self.compute_point_analytics(slow_pt, Input)
                k = np.sign(np.dot(r, direction))
                l *= k; r *= k

                slow_points.append(deepcopy(slow_pt))
                fun_vals.append(fun_val)
                jacs.append(deepcopy(J))
                eigs.append(deepcopy(E))
                selection_vects.append(deepcopy(l))
                principle_eigenvects.append(deepcopy(r))
            selection_vects = make_orientation_consistent(selection_vects)
            principle_eigenvects = make_orientation_consistent(principle_eigenvects)

            self.data[context]["slow_points"] = (np.array(slow_points))
            self.data[context]["fun_val"] = (np.array(fun_vals))
            self.data[context]["jac"] = (np.array(jacs))
            self.data[context]["eigs"] = (np.array(eigs))
            self.data[context]["l"] = (np.array(selection_vects))
            self.data[context]["r"] = (np.array(principle_eigenvects))
        return deepcopy(self.data)

    def plot_fp_2D_CDDM(self,
                        patience=10,
                        fun_tol=1e-12,
                        stop_length=20,
                        sigma_init_guess=10,
                        eig_cutoff=1e-10,
                        diff_cutoff=1e-7):
        '''
        same params as in 'get_fixed_points'
        :return: fig of projected FPs on a subspace spanned by choice and context axes
        '''
        default_input = np.array([0.0, 0.0, 0.5, 0.5, 0.5, 0.5])
        P = np.hstack([self.choice_axis.reshape(-1, 1), self.context_axis.reshape(-1, 1)])
        data_to_plot = {}; data_to_plot["motion"] = {}; data_to_plot["color"] = {}

        for p, context in enumerate(["motion", "color"]):
            ctxt_ind = 0 if context == 'motion' else 1
            Input = deepcopy(default_input)
            Input[ctxt_ind] = 1
            # if the FPs are not yet calculated
            if (len(list(self.data[context].keys())) == 0):
                self.calc_fixed_points_CDDM(patience=patience,
                                           fun_tol=fun_tol,
                                           stop_length=stop_length,
                                           sigma_init_guess=sigma_init_guess,
                                           eig_cutoff=eig_cutoff,
                                           diff_cutoff=diff_cutoff)
            # projecting fixed points onto 2D space
            for key in ["stable_fps", "unstable_fps", "marginally_stable_fps"]:
                if self.data[context][key].shape[0] != 0:
                    data_to_plot[context][key] = self.data[context][key] @ P

        # plotting
        fig = plt.figure(figsize=(6, 6))
        fig.suptitle(r"Fixed points projected on (choice, context)", fontsize=16)
        markers = ["o", "x", "o"]; colors = ["blue", "red", "k"]
        for context in ["motion", "color"]:
            for k, key in enumerate(["stable_fps", "unstable_fps", "marginally_stable_fps"]):
                if self.data[context][key].shape[0] != 0:
                    for i in range(data_to_plot[context][key].shape[0]):
                        plt.scatter(data_to_plot[context][key][i, 0],
                                    data_to_plot[context][key][i, 1],
                                    marker=markers[k], s=100, color=colors[k])
        plt.ylabel("Context", fontsize=16)
        plt.xlabel("Choice", fontsize=16)
        plt.grid(True)
        return fig

    def plot_LineAttractor_3D(self,
                              N_points=31,
                              patience=30,
                              fun_tol=1e-12,
                              stop_length=20,
                              sigma_init_guess=10,
                              eig_cutoff=1e-10,
                              diff_cutoff=1e-7,
                              nudge=0.05,
                              obj_max_iter=100,
                              T_steps=1000,
                              relax_steps=10,
                              steps_stim_on=500,
                              steps_context_only_on=250):
        '''
        Plots
        Figure with projected line attractors within each context on a (sensory, context, choice)-subspace
        also plots 6 trajectories: three in each context.
        light blue - color context, orange - motion context,
        solid lines - trajectories with the decision to the right, dashed lines - to the left.
        two magenta trajectores - trials within each context with 0 coherence
        :return:
        '''
        nDim = 3
        # projection matrix
        P_matrix = np.zeros((self.RNN.N, 3))
        P_matrix[:, 0] = self.choice_axis
        P_matrix[:, 1] = self.context_axis
        P_matrix[:, 2] = self.sensory_axis
        if not ("slow_points" in (list(self.data["motion"].keys()))):
            data_dict = self.calc_LineAttractor_analytics(N_points=N_points,
                                                          patience=patience,
                                                          fun_tol=fun_tol,
                                                          stop_length=stop_length,
                                                          sigma_init_guess=sigma_init_guess,
                                                          eig_cutoff=eig_cutoff,
                                                          diff_cutoff=diff_cutoff,
                                                          obj_max_iter=obj_max_iter,
                                                          T_steps=T_steps,
                                                          relax_steps=relax_steps)
        else:
            data_dict = deepcopy(self.data)

        trajectories = dict()
        trajectories["motion"] = {}
        trajectories["color"] = {}
        for ctxt in ["motion", "color"]:
            trajectories[ctxt] = {}
            for stim_status in ["relevant", "irrelevant"]:
                trajectories[ctxt][stim_status] = {}
                for period in ["context_only_on", "stim_on", "stim_off"]:
                    trajectories[ctxt][stim_status][period] = {}

        colors, cmp = get_colormaps()
        red, blue, bluish, green, orange, lblue, violet = colors
        colors_trajectories = dict()
        colors_trajectories["motion"] = dict()
        colors_trajectories["color"] = dict()
        colors_trajectories["motion"]["relevant"] = colors[5]
        colors_trajectories["motion"]["irrelevant"] = colors[3]
        colors_trajectories["color"]["relevant"] = colors[1]
        colors_trajectories["color"]["irrelevant"] = colors[3]

        for ctxt in ["motion", "color"]:
            val = 1 if ctxt == 'motion' else 0
            for stim_status in ["relevant", "irrelevant"]:
                self.RNN.clear_history()
                rel_inds = [2, 3] if ctxt == 'motion' else [4, 5]
                irrel_inds = [4, 5] if ctxt == 'motion' else [2, 3]
                nudge_inds = rel_inds if stim_status == 'relevant' else irrel_inds

                x0 = 0.00 * np.random.randn(self.RNN.N)
                input = np.array([val, 1 - val, 0.0, 0.0, 0.0, 0.0])
                input_timeseries = np.repeat(input[:, np.newaxis], axis=1, repeats=steps_context_only_on)
                self.RNN.y = deepcopy(x0)
                self.RNN.run(input_timeseries=input_timeseries, save_history=True)
                x_trajectory_context_only_on = self.RNN.get_history()
                trajectories[ctxt][stim_status]["context_only_on"] = x_trajectory_context_only_on
                self.RNN.clear_history()

                x0 = deepcopy(x_trajectory_context_only_on[-1, :])
                for direction in ['left', 'right', 'center']:
                    input = deepcopy(np.array([val, 1 - val, 0.5, 0.5, 0.5, 0.5]))
                    if direction == 'left':
                        input[nudge_inds] -= np.array([nudge, -nudge])
                    elif direction == 'right':
                        input[nudge_inds] += np.array([nudge, -nudge])
                    input_timeseries = np.repeat(input[:, np.newaxis], axis=1, repeats=steps_stim_on)
                    self.RNN.y = deepcopy(x0)
                    self.RNN.run(input_timeseries=input_timeseries, save_history=True)
                    trajectory = self.RNN.get_history()
                    trajectories[ctxt][stim_status]["stim_on"][direction] = deepcopy(trajectory)
                    self.RNN.clear_history()

        colors_trajectories = dict()
        colors_trajectories["motion"] = dict()
        colors_trajectories["color"] = dict()
        colors_trajectories["motion"]["relevant"] = orange
        colors_trajectories["motion"]["irrelevant"] = orange
        colors_trajectories["color"]["relevant"] = lblue
        colors_trajectories["color"]["irrelevant"] = lblue
        colors_LA = dict()
        colors_LA["motion"] = bluish
        colors_LA["color"] = green

        fig_3D = plt.figure(figsize=(7, 7))
        ax = fig_3D.add_subplot(projection='3d')
        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.set_xlabel("Choice", fontsize=20)
        ax.set_ylabel("Context", fontsize=20)
        ax.set_zlabel("Sensory", fontsize=20)

        # initial point trajectory
        ax.scatter([0, 0], [0, 0], [0, 0], color='r', marker='o', s=10, alpha=0.9)

        for ctxt in ["motion", "color"]:
            slow_points_projected = data_dict[ctxt]["slow_points"] @ P_matrix
            ax.scatter(*(slow_points_projected[:, k] for k in range(nDim)), color=colors_LA[ctxt], marker='o', s=6, alpha=0.2)
            ax.plot(*(slow_points_projected[:, k] for k in range(nDim)), color=colors_LA[ctxt])
            for stim_status in ["relevant"]:
                clr = colors_trajectories[ctxt][stim_status]

                trajectory_projected = trajectories[ctxt][stim_status]["context_only_on"] @ P_matrix
                ax.plot(*(trajectory_projected[:, t] for t in range(nDim)),
                        linestyle='-', linewidth=1.5, color=clr, alpha=0.8)

                linestyles = ['-', ':', '-']
                colors = [clr, clr, 'm']
                for k, key in enumerate(["right", "left", "center"]):
                    trajectory_projected = trajectories[ctxt][stim_status]["stim_on"][key] @ P_matrix
                    ax.plot(*(trajectory_projected[:, t] for t in range(nDim)),
                            linestyle=linestyles[k], linewidth=3, color=colors[k], alpha=0.8)

        fig_3D.subplots_adjust()
        ax.view_init(12, 228)
        fig_3D.subplots_adjust()
        plt.tight_layout()
        return fig_3D

    def plot_RHS_over_LA(self,
                         N_points=31,
                         patience=30,
                         fun_tol=1e-12,
                         stop_length=20,
                         sigma_init_guess=10,
                         eig_cutoff=1e-10,
                         diff_cutoff=1e-7,
                         obj_max_iter=100,
                         T_steps=1000,
                         relax_steps=10):
        if not ("slow_points" in (list(self.data["motion"].keys()))):
            data_dict = self.calc_LineAttractor_analytics(N_points=N_points,
                                                          patience=patience,
                                                          fun_tol=fun_tol,
                                                          stop_length=stop_length,
                                                          sigma_init_guess=sigma_init_guess,
                                                          eig_cutoff=eig_cutoff,
                                                          diff_cutoff=diff_cutoff,
                                                          obj_max_iter=obj_max_iter,
                                                          T_steps=T_steps,
                                                          relax_steps=relax_steps)
        else:
            data_dict = deepcopy(self.data)
        colors, cmp = get_colormaps()
        red, blue, bluish, green, orange, lblue, violet = colors
        fig_RHS = plt.figure(figsize=(12, 3))
        plt.suptitle(r"$\||RHS(x)\||^2$", fontsize = 16)
        plt.axhline(0, color="gray", linewidth=2, alpha=0.2)
        x = np.linspace(0, 1, N_points)
        plt.plot(x, np.array(data_dict["motion"]["fun_val"]), color=bluish, linewidth=3, linestyle='-', label="motion")
        plt.plot(x, np.array(data_dict["color"]["fun_val"]), color=green, linewidth=3, linestyle='-', label="color")
        plt.legend(fontsize=14)
        plt.xlabel("distance along the LA", fontsize=16)
        plt.ylabel(r"$\||RHS(x)\||$", fontsize=16)
        plt.grid(True)
        return fig_RHS