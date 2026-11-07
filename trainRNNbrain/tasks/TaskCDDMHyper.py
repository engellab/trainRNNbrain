import numpy as np
from trainRNNbrain.tasks.TaskCDDM import TaskCDDM


class TaskCDDMHyper(TaskCDDM):
    """CDDM with a read-out of every product of the trial's two binary signs, plus the cued choice:
    the CDDM counterpart of TaskNBitFlipFlopHyper, the new task of the B4 transfer test.

    The hyper flip-flop keeps the flip-flop's inputs and outputs every product of the bit states over
    the non-empty subsets of bits (the original single bits among them). A CDDM trial has two binary
    sensory signs, m = sign(motion coherence) and c = sign(colour coherence), and the context; the
    products over the non-empty subsets of {m, c} are m, c and m * c. So the inputs, the trial
    structure and the condition grid are TaskCDDM's, unchanged (6 inputs: 2 context cues, motion
    right/left, colour right/left), and the target has 5 channels in the decision window
    [dec_on - 1, dec_off), 0 elsewhere:
      0, 1   the cued choice, exactly TaskCDDM's 2-output target (one-hot right/left by the sign of
             the cued coherence; 0.5 on both when it is 0): the original task, as the hyper
             flip-flop's single-bit channels are the flip-flop's
      2      m, the motion sign whatever the cue (+1, -1, or 0 at zero coherence)
      3      c, the colour sign whatever the cue
      4      m * c, +1 when the two signs agree, -1 when they differ, 0 if either coherence is 0
    The three new channels use the +-1 coding of the hyper flip-flop and of TaskCDDM's 1-output
    target. n_outputs must be 5.
    """

    def __init__(self, n_steps, n_inputs, n_outputs,
                 cue_on, cue_off, stim_on, stim_off, dec_on, dec_off,
                 coherences, seed=None):
        """Same arguments as TaskCDDM; n_outputs is checked against 5 (2 cued-choice channels + m, c, m*c)."""
        if n_outputs != 5:
            raise ValueError(f"TaskCDDMHyper has 5 outputs (cued choice x 2, motion sign, colour sign, "
                             f"their product), got n_outputs={n_outputs}")
        TaskCDDM.__init__(self, n_steps, n_inputs, n_outputs, cue_on, cue_off, stim_on, stim_off,
                          dec_on, dec_off, coherences, seed)

    def generate_input_target_stream(self, context, motion_coh, color_coh):
        """One trial: TaskCDDM's input stream and the 5-channel target described in the class docstring.

        Args:
            context: "motion" or "color"; motion_coh, color_coh: float in [-1, 1].
        Returns:
            (input_stream (n_inputs, n_steps), target_stream (5, n_steps)).
        """
        input_stream, target_stream = TaskCDDM.generate_input_target_stream(self, context, motion_coh, color_coh)
        # TaskCDDM fills every one of n_outputs rows by its 2-output rule; rows 0-1 are that rule's,
        # rows 2-4 are rewritten here
        target_stream[2:] = 0.0
        m, c = np.sign(motion_coh), np.sign(color_coh)
        window = slice(self.dec_on - 1, self.dec_off)
        target_stream[2, window] = m
        target_stream[3, window] = c
        target_stream[4, window] = m * c
        return input_stream, target_stream
