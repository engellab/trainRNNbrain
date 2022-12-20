import sys
import os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, '../')
from tqdm.auto import tqdm
from scipy.optimize import fsolve, minimize
from src.RNN_numpy import RNN_numpy
from src.utils import *
from copy import deepcopy

def ReLU(x):
    return np.maximum(x, 0)

def rhs(x, W_rec, W_inp, bias_rec, Input):
    return (-x + ReLU((W_rec @ x.reshape(-1, 1)).flatten() + (W_inp @ Input.reshape(-1, 1)).flatten() + bias_rec.flatten()))

def rhs_jac(x, W_rec, W_inp, bias_rec, Input):
    N = W_rec.shape[0]
    arg = ((W_rec @ x.reshape(-1, 1)).flatten() + (W_inp @ Input.reshape(-1, 1)).flatten() + bias_rec.flatten())
    m = 0.5
    D = np.diag(np.heaviside(arg, m))
    return -np.eye(N) + W_rec @ D

def objective(x, W_rec, W_inp, bias_rec, Input):
    # while finding the fixed point minimize the norm of the right-hand side of the dynamical equations
    return np.sum((rhs(x, W_rec, W_inp, bias_rec, Input)) ** 2)

def objecive_grad(x, W_rec, W_inp, bias_rec, Input):
    return 2 * (rhs(x, W_rec, W_inp, bias_rec, Input).reshape(1, -1) @ rhs_jac(x, W_rec, W_inp, bias_rec, Input)).flatten()

def step(dt, tau, x,  W_rec, W_inp, bias_rec, Input):
    return x + (dt/tau) * (rhs(x, W_rec, W_inp, bias_rec, Input))

def run(steps, dt, tau, x, W_rec, W_inp, bias_rec, Input):
    xs = []
    xs.append(x)
    for i in range(steps):
        x = step(dt, tau, x, W_rec, W_inp, bias_rec, Input)
        xs.append(deepcopy(x))
    xs = np.array(xs)
    return xs

def find_fixed_points(W_inp, W_rec, bias_rec, Input,
                      fun_tol=1e-12, patience=100, stop_length=20, sigma_init_guess=10, eig_cutoff=1e-10):
    # fun_tol: only consider the points such that the ||RHS(x)|| < fun_tol
    unstable_fps = []
    stable_fps = []
    marginally_stable_fps = []
    N = W_rec.shape[0]
    cntr = 0
    while (cntr <= patience) and (len(unstable_fps) < stop_length):
        x0 = sigma_init_guess * np.random.randn(N)
        x_root = fsolve(rhs, x0, args=(W_rec, W_inp, bias_rec, Input,))
        fun = objective(x_root, W_rec, W_inp, bias_rec, Input)
        if fun < fun_tol:
            J = rhs_jac(x_root, W_rec, W_inp, bias_rec, Input)
            L = np.linalg.eigvals(J)
            L_0 = np.max(np.real(L))
            cntr += 1
            if (np.abs(L_0) <= eig_cutoff) and (not in_the_list(x_root, marginally_stable_fps, cutoff_diff=1e-8)):
                if (len(marginally_stable_fps) <= stop_length):
                    marginally_stable_fps.append(x_root)
                    cntr=0
            elif (L_0 > eig_cutoff) and (not in_the_list(x_root, unstable_fps, cutoff_diff=1e-8)):
                if (len(unstable_fps) <= stop_length):
                    unstable_fps.append(x_root)
                    cntr = 0
            elif (L_0 < -eig_cutoff) and (not in_the_list(x_root, stable_fps, cutoff_diff=1e-8)):
                if (len(stable_fps) <= stop_length):
                    stable_fps.append(x_root)
                    cntr = 0
    unstable_fps = np.array(unstable_fps)
    stable_fps = np.array(stable_fps)
    marginally_stable_fps = np.array(marginally_stable_fps)
    return unstable_fps, stable_fps, marginally_stable_fps


def find_end_points_of_line_attractor(N, dt, tau,
                                      W_inp, W_rec, W_out, bias_rec, bias_out, y_init,
                                      inp_coeffs, nudge=0.05, T_steps=1000, relax_steps=100):
    '''
    to get the line attractor, we need to get the leftmost and the rightmost point
    '''
    RNN = RNN_numpy(N=N, dt=dt, tau=tau,
                  W_inp=W_inp, W_rec=W_rec, W_out=W_out,
                  bias_rec=bias_rec, bias_out=bias_out)
    RNN.y = deepcopy(y_init)
    context = 'motion' if inp_coeffs[0] == 1 else 'color'
    ctxt_ind = 0 if context == 'motion' else 1
    default_inp_coeffs = np.array([0, 0, 0.5, 0.5, 0.5, 0.5])
    default_inp_coeffs[ctxt_ind] = 1
    bias_inp_coeff = np.array([nudge, -nudge])  # to favour a decision either to the right or to the left
    input_coeffs_right_decision = deepcopy(default_inp_coeffs)
    input_coeffs_left_decision = deepcopy(default_inp_coeffs)
    if context == 'motion':
        inds = [2, 3]
        input_coeffs_right_decision[inds] += bias_inp_coeff
        input_coeffs_left_decision[inds] -= bias_inp_coeff
    elif context == 'color':
        inds = [4, 5]
        input_coeffs_right_decision[inds] += bias_inp_coeff
        input_coeffs_left_decision[inds] -= bias_inp_coeff

    # find right point
    RNN.y_init = deepcopy(y_init)
    RNN.y = deepcopy(y_init)
    RNN.run(num_steps=T_steps, Inputs=np.vstack([input_coeffs_right_decision.reshape(1, -1)] * T_steps))
    RNN.run(num_steps=relax_steps, Inputs=np.vstack([default_inp_coeffs.reshape(1, -1)] * relax_steps))
    right_point = deepcopy(RNN.y)

    # find left point
    RNN.y_init = deepcopy(y_init)
    RNN.y = deepcopy(y_init)
    RNN.clear_history()
    RNN.run(num_steps=T_steps, Inputs=np.vstack([input_coeffs_left_decision.reshape(1, -1)] * T_steps))
    RNN.run(num_steps=relax_steps, Inputs=np.vstack([default_inp_coeffs.reshape(1, -1)] * relax_steps))
    left_point = deepcopy(RNN.y)
    return left_point, right_point

def find_slow_points(N_points,
                     W_rec, W_inp, W_out, bias_rec, bias_out, dt, tau,
                     inp_coeffs,
                     y_init=None,
                     nudge=0.05,
                     T_steps=2000,
                     relax_steps=200,
                     fun_tol=1e-6,
                     patience=100,
                     stop_length=100,
                     maxiter=1000):
    if bias_rec is None:
        bias_rec = np.zeros(W_rec.shape[0])
    if bias_out is None:
        bias_out = 0
    # "nudge" defines the bias of choice to the right or to the left given the input where the
    # sensory evidence to the right is equal to sensory evidence to the left
    N = W_rec.shape[0]
    unstable_fps, stable_fps, marginally_stable_fps = find_fixed_points(W_inp=W_inp,
                                                                        W_rec=W_rec,
                                                                        bias_rec=bias_rec,
                                                                        Input=inp_coeffs,
                                                                        fun_tol=fun_tol,
                                                                        patience=patience,
                                                                        stop_length=stop_length)

    slow_points = []
    data_dict = {}
    if y_init is None:
        y_init = np.zeros(N)

    left_point, right_point = find_end_points_of_line_attractor(N=N, dt=dt, tau=tau,
                                                                W_inp=W_inp, W_rec=W_rec, W_out=W_out,
                                                                bias_rec=bias_rec, bias_out=bias_out,
                                                                y_init=y_init,
                                                                inp_coeffs=inp_coeffs,
                                                                nudge=nudge,
                                                                T_steps=T_steps,
                                                                relax_steps=relax_steps)
    x_init = deepcopy(left_point)
    #define the starting direction for search
    increment = (1 / (N_points-1)) * (right_point - left_point)

    for i in tqdm(range(N_points)):
        #minimize ||RHS(x)|| such that the x stays within a space orthogonal to the line attractor
        res = minimize(objective, x0=x_init, args=(W_rec, W_inp, bias_rec, inp_coeffs,), method='SLSQP',
                       jac=objecive_grad, options={'disp': False, 'maxiter': maxiter},
                       constraints={'type': 'eq', 'fun': lambda x: np.dot(x - x_init, increment)})
        x_root = deepcopy(res.x)
        x_init = x_root + increment
        slow_points.append(deepcopy(x_root))
        x_prev = deepcopy(x_root)
        data_dict["slow_points"] = np.array(deepcopy(slow_points))
        data_dict["stable_fp"] = np.array(deepcopy(stable_fps))
        data_dict["unstable_fp"] = np.array(deepcopy(unstable_fps))
        data_dict["marginally_stable_fp"] = np.array(deepcopy(marginally_stable_fps))
    return data_dict


def make_orientation_consistent(vectors, num_iter=10):
    for i in range(num_iter): # np.min(dot_prod) < 0:
        average_vect = np.mean(vectors, axis=0)
        average_vect /= np.linalg.norm(average_vect)
        dot_prod = vectors @ average_vect
        vectors[np.where(dot_prod < 0)[0], :] *= -1
    return vectors

def get_LA_analytics(net, N_points=31, T_steps=750, relax_steps=10, nudge=0.01,
                     stop_length=100, patience=100, fun_tol=1e-10):
    inp_coeffs = [np.array([1.0, 0.0, 0.5, 0.5, 0.5, 0.5]), np.array([0.0, 1.0, 0.5, 0.5, 0.5, 0.5])]
    net_slow_points_data = {}
    net_slow_points_data["motion"] = {}
    net_slow_points_data["color"] = {}
    net_slow_points_data["motion"]["slow_points"] = []
    net_slow_points_data["motion"]["fun_val"] = []
    net_slow_points_data["motion"]["Jac"] = []
    net_slow_points_data["motion"]["Eigs"] = []
    net_slow_points_data["motion"]["l"] = []

    net_slow_points_data["color"]["slow_points"] = []
    net_slow_points_data["color"]["fun_val"] = []
    net_slow_points_data["color"]["Jac"] = []
    net_slow_points_data["color"]["Eigs"] = []
    net_slow_points_data["color"]["l"] = []

    choice_axis = net.W_out.flatten() if net.W_out.shape[0] == 1 else (net.W_out[0, :] - net.W_out[1, :])
    for j, ctxt in enumerate(["motion", "color"]):
        val = 1 if ctxt == "motion" else 0
        input = np.array([val, 1 - val, 0.5, 0.5, 0.5, 0.5])
        slow_points_data_ctx = find_slow_points(N_points=N_points,
                                                W_rec=net.W_rec, W_inp=net.W_inp, W_out=net.W_out,
                                                bias_rec=net.bias_rec, bias_out=net.bias_out,
                                                dt=net.dt, tau=net.tau,
                                                inp_coeffs=input,
                                                y_init=net.y_init,
                                                T_steps=T_steps, relax_steps=relax_steps,
                                                nudge=nudge,
                                                stop_length=stop_length,
                                                fun_tol=fun_tol,
                                                patience=patience)
        slow_points_data_ctx = slow_points_data_ctx["slow_points"]
        direction = slow_points_data_ctx[-1, :] - slow_points_data_ctx[0, :]
        direction *= np.sign(np.dot(choice_axis, direction))
        fun_vals = []

        for i in range(slow_points_data_ctx.shape[0]):
            params = {"W_rec": net.W_rec, "W_inp": net.W_inp,
                      "bias_rec": net.bias_rec,
                      "Input": np.array([val, 1 - val, 0.5, 0.5, 0.5, 0.5]).reshape(-1, 1)}
            slow_pt = slow_points_data_ctx[i, :]
            net.y = deepcopy(slow_pt)
            J = net.rhs_jac(input)
            E, R = np.linalg.eig(J)
            E, R = sort_eigs(E, R)
            L = np.linalg.inv(R)

            l = np.real(L[0, :])
            r = np.real(R[:, 0])
            k = np.sign(np.dot(r, direction))
            l *= k;
            r *= k

            net_slow_points_data[ctxt]["slow_points"].append(deepcopy(slow_pt))
            net_slow_points_data[ctxt]["fun_val"].append((objective(slow_pt, **params)))
            net_slow_points_data[ctxt]["Jac"].append(deepcopy(J))
            net_slow_points_data[ctxt]["Eigs"].append(deepcopy(E))
            net_slow_points_data[ctxt]["l"].append(deepcopy(l))
        ls_array = np.stack(net_slow_points_data[ctxt]["l"])
        ls_array = make_orientation_consistent(ls_array)
        net_slow_points_data[ctxt]["l"] = [ls_array[i, :] for i in range(ls_array.shape[0])]

    return net_slow_points_data