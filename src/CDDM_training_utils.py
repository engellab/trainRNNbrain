import sys
sys.path.insert(0, "../")
import numpy as np
from copy import deepcopy
import torch
from src.generate_CDDM_trials import generate_all_trials

def L2_ortho(rnn, X = None, y = None):
    # regularization of the input and ouput matrices
    b = torch.cat((rnn.input_layer.weight, rnn.output_layer.weight.t()), dim=1)
    b = b / torch.norm(b, dim=0)
    return torch.norm(b.t() @ b - torch.diag(torch.diag(b.t() @ b)), p=2)

def train_step(rnn, input, target_output, mask, optimizer, loss_fn):
    states, predicted_output = rnn(input)
    loss = loss_fn(target_output[:, mask, :], predicted_output[:, mask, :]) + rnn.lambda_o * L2_ortho(rnn)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    error_vect = torch.sum(((target_output[:, mask, :] - predicted_output[:, mask, :]) ** 2).squeeze(), dim=1) / len(mask)
    return loss.item(), error_vect

def eval_step(rnn, input, target_output, mask, loss_fn):
    with torch.no_grad():
        rnn.eval()
        states, predicted_output_val = rnn(input, w_noise=False)
        val_loss = loss_fn(target_output[:, mask, :],
                             predicted_output_val[:, mask, :]) + rnn.lambda_o * L2_ortho(rnn)
        return float(val_loss.numpy())

def print_iteration_info(iter, train_loss, min_train_loss, val_loss, min_val_loss):
    gr_prfx = '\033[92m'
    gr_sfx = '\033[0m'

    train_prfx = gr_prfx if (train_loss <= min_train_loss) else ''
    train_sfx = gr_sfx if (train_loss <= min_train_loss) else ''
    if not (val_loss is None):
        val_prfx = gr_prfx if (val_loss <= min_val_loss) else ''
        val_sfx = gr_sfx if (val_loss <= min_val_loss) else ''
        print(f"iteration {iter},"
              f" train loss: {train_prfx}{np.round(train_loss, 6)}{train_sfx},"
              f" validation loss: {val_prfx}{np.round(val_loss, 6)}{val_sfx}")
    else:
        print(f"iteration {iter},"
              f" train loss: {train_prfx}{np.round(train_loss, 6)}{train_sfx}")

def train_CDDM(rnn,
               criterion,
               optimizer,
               max_iter,
               tol,
               data_gen_params,
               train_mask,
               constrained=True,
               generator_numpy=None,
               run_instance=None):
    train_losses = []
    val_losses = []
    rnn.train()
    min_train_loss = np.inf
    min_val_loss = np.inf

    coherences_train = data_gen_params["coherences_train"]
    coherences_valid = data_gen_params["coherences_valid"]
    # remove from data gen params to use the dictionary further
    data_gen_params.pop("coherences_train")
    data_gen_params.pop("coherences_valid")

    for iter in range(max_iter):
        input_batch, target_batch, conditions_batch = generate_all_trials(**data_gen_params,
                                                                          coherences=coherences_train,
                                                                          generator_numpy=generator_numpy)
        input_batch = torch.from_numpy(input_batch.astype("float32")).to(rnn.device)
        target_batch = torch.from_numpy(target_batch.astype("float32")).to(rnn.device)

        train_loss, error_vect = train_step(rnn,
                                            input=input_batch,
                                            target_output=target_batch,
                                            mask=train_mask,
                                            optimizer=optimizer,
                                            loss_fn=criterion)
        if constrained:
            rnn.output_layer.weight.data = torch.maximum(rnn.output_layer.weight.data, torch.tensor(0))
            rnn.input_layer.weight.data = torch.maximum(rnn.input_layer.weight.data, torch.tensor(0))
            # Dale's law
            rnn.recurrent_layer.weight.data = torch.maximum(rnn.recurrent_layer.weight.data * rnn.dale_mask, torch.tensor(0)) * rnn.dale_mask

        #validation
        input_val, target_output_val, conditions_val = generate_all_trials(**data_gen_params,
                                                                           coherences=coherences_valid,
                                                                           generator_numpy=generator_numpy)
        input_val = torch.from_numpy(input_val.astype("float32")).to(rnn.device)
        target_output_val = torch.from_numpy(target_output_val.astype("float32")).to(rnn.device)
        val_loss = eval_step(rnn, input_val, target_output_val, train_mask, criterion)
        #keeping track of train and valid losses and printing
        print_iteration_info(iter, train_loss, min_train_loss, val_loss, min_val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            best_net_params = deepcopy(rnn.get_params())
        if train_loss <= min_train_loss:
            min_train_loss = train_loss

        if not (run_instance is None):
            run_instance["train_loss"].log(train_loss)
            run_instance["val_loss"].log(val_loss)

        if val_loss <= tol:
            set_params(rnn, best_net_params)
            return rnn, train_losses, val_losses, best_net_params

    set_params(rnn, best_net_params)
    return rnn, train_losses, val_losses, best_net_params

def set_params(rnn, params):
    rnn.output_layer.weight.data = torch.from_numpy(params["W_out"])
    rnn.input_layer.weight.data = torch.from_numpy(params["W_inp"])
    rnn.recurrent_layer.weight.data = torch.from_numpy(params["W_rec"])
    if not (rnn.recurrent_layer.bias is None):
        rnn.recurrent_layer.bias.data = torch.from_numpy(params["bias_rec"])
    if not (rnn.output_layer.bias is None):
        rnn.output_layer.bias.data = torch.from_numpy(params["bias_out"])
    rnn.y_init = torch.from_numpy(params["y_init"])
    return rnn

def trained_net_validation(RNN_valid, data_gen_params, mask, sigma_rec, sigma_inp, generator_numpy=None):
    input_val, target_output_val, conditions_val = generate_all_trials(n_steps=data_gen_params["n_steps"],
                                                                     coherences=data_gen_params["coherences_valid"],
                                                                     protocol_dict=data_gen_params["protocol_dict"],
                                                                     num_outputs=data_gen_params["num_outputs"],
                                                                     generator_numpy=generator_numpy)

    MSE_output = []
    trial_length = input_val.shape[1]
    for i in range(len(input_val)):
        RNN_valid.y = deepcopy(RNN_valid.y_init)
        RNN_valid.clear_history()
        RNN_valid.run(num_steps=trial_length, Inputs=input_val[i], sigma_rec=sigma_rec, sigma_inp=sigma_inp, save_history=True, generator_numpy=generator_numpy)
        output_prediction = RNN_valid.get_output()
        output_target = target_output_val[i]
        MSE_output.append(np.mean((output_prediction[mask, :] - output_target[mask, :]) ** 2))
    MSE_score = np.round(np.mean(MSE_output), 7)
    return MSE_score


