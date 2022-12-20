import numpy as np
from copy import deepcopy
'''
Generate the input time-series and the correct output for the CDDM task for multiple coherences
'''

def generate_input_target_stream(context, motion_coh, color_coh,
                                 protocol_dict,
                                 n_steps=750,
                                 num_outputs=2):
    '''
    generate an input and target for a single trial with the supplied coherences
    :param context: could be either 'motion' or 'color' (see Mante et. all 2013 paper)
    :param motion_coh: coherence of information in motion channel, range: (0, 1)
    :param color_coh: coherence of information in color channel, range: (0, 1)
    :param n_steps: number of steps in the trial, default is 750
    :param protocol_dict: a dictionary which provides the trial structure:
        cue_on, cue_off - defines the timespan when the contextual information is supplied
        stim_on, stim_off - defines the timespan when the sensory information is supplied
        dec_on, dec_off - defines the timespan when the decision has to be present in the target stream
        all the values should be less than n_steps

    :param num_outputs: number of target outputs supplied.
        in case of num_outputs = 1, decision is reflected in the single output
    :return: input_stream, target_stream
    input_stream - input time series (both context and sensory): num_batch x num_steps x 6
    target_stream - time sereis reflecting the correct decision: num_batch x num_steps x num_outputs
    '''

    cue_on = protocol_dict["cue_on"]
    cue_off = protocol_dict["cue_off"]
    stim_on = protocol_dict["stim_on"]
    stim_off = protocol_dict["stim_off"]
    dec_on = protocol_dict["dec_on"]
    dec_off = protocol_dict["dec_off"]

    # given the context and coherences of signals
    # generate input array (n_steps x 6)
    # and target array (ideal output of the Decision-making system)

    # Transform coherence to signal
    motion_r = (1 + motion_coh) / 2
    motion_l = 1 - motion_r
    color_r = (1 + color_coh) / 2
    color_l = 1 - color_r

    trial_length = n_steps

    # Cue input stream
    cue_input = np.zeros([trial_length, 6])
    if context == "motion":
        cue_input[cue_on:cue_off, 0] = 1 * np.ones([cue_off - cue_on, 1]).squeeze()
    else:
        cue_input[cue_on:cue_off, 1] = 1 * np.ones([cue_off - cue_on, 1]).squeeze()

    sensory_input = np.zeros([trial_length, 6])
    # Motion input stream
    sensory_input[stim_on - 1:stim_off, 2] = motion_r * np.ones([stim_off - stim_on + 1])
    sensory_input[stim_on - 1:stim_off, 3] = motion_l * np.ones([stim_off - stim_on + 1])
    # Color input stream
    sensory_input[stim_on - 1:stim_off, 4] = color_r * np.ones([stim_off - stim_on + 1])
    sensory_input[stim_on - 1:stim_off, 5] = color_l * np.ones([stim_off - stim_on + 1])

    input_stream = cue_input + sensory_input

    # Target stream
    if num_outputs == 1:
        target_stream = np.zeros((trial_length, 1))
        target_stream[dec_on - 1:dec_off, 0] = np.sign(motion_coh) if (context == 'motion') else np.sign(color_coh)
    elif num_outputs == 2:
        target_stream = np.zeros((trial_length, 2))
        relevant_coh = motion_coh if (context == 'motion') else color_coh
        if relevant_coh == 0.0:
            pass
        else:
            decision = np.sign(relevant_coh)
            ind = 0 if (decision == 1.0) else 1
            target_stream[dec_on - 1:dec_off, ind] = 1
    return input_stream, target_stream


def generate_all_trials(n_steps,
                        coherences,
                        protocol_dict,
                        num_outputs=2,
                        shuffle=False,
                        generator_numpy=None):
    '''
    See above, generate_input_target_stream
    :param shuffle: shuffle the final array
    :param generator_numpy: the random generator (for reproducibility, if using shuffle=True)
    :return: array of inputs, array of targets, and the conditions (context, coherences and the correct choice)
    '''

    inputs = []
    targets = []
    conditions = []
    if generator_numpy is None:
        generator_numpy = np.random.default_rng()
    for context in ["motion", "color"]:
        for c1 in coherences:
            for c2 in coherences:
                relevant_coh = c1 if context == 'motion' else c2
                irrelevant_coh = c2 if context == 'motion' else c1
                motion_coh = c1 if context == 'motion' else c2
                color_coh = c1 if context == 'color' else c2
                coh_pair = (relevant_coh, irrelevant_coh)

                correct_choice = 1 if ((context == "motion" and motion_coh > 0) or (context == "color" and color_coh > 0)) else -1
                conditions.append({'context': context,
                                   'motion_coh': motion_coh,
                                   'color_coh': color_coh,
                                   'correct_choice': correct_choice})
                input_stream, target_stream = generate_input_target_stream(context, coh_pair[0], coh_pair[1], n_steps=n_steps,
                                                                           protocol_dict=protocol_dict,
                                                                           num_outputs=num_outputs)
                inputs.append(deepcopy(input_stream))
                targets.append(deepcopy(target_stream))

    inputs = np.stack(inputs, axis=0)
    targets = np.stack(targets, axis=0)
    if shuffle:
        perm = generator_numpy.permutation(len(inputs))
        inputs = inputs[perm, :, :]
        targets = targets[perm, :, :]
        conditions = [conditions[index] for index in perm]
    return inputs, targets, conditions


if __name__ == '__main__':
    coherences = np.linspace(-1, 1, 11)
    n_steps = 750
    protocol_dict = {"cue_on": 0, "cue_off": n_steps,
                     "stim_on": int(n_steps//3), "stim_off": n_steps,
                     "dec_on": int(2 * n_steps//3), "dec_off": n_steps}
    input_batch, target_batch, conditions_batch = generate_all_trials(n_steps=n_steps,
                                                                      coherences=coherences,
                                                                      protocol_dict=protocol_dict,
                                                                      num_outputs=2)