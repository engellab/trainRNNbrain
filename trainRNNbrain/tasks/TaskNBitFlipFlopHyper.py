import itertools
import numpy as np
from trainRNNbrain.tasks.TaskNBitFlipFlop import TaskNBitFlipFlop


class TaskNBitFlipFlopHyper(TaskNBitFlipFlop):
    """k-bit flip-flop whose read-out is EVERY product of bit states: 2^k - 1 output channels.

    Inputs and bit dynamics are exactly TaskNBitFlipFlop's (k independent Poisson-timed +-1 pulse
    trains, each bit holding the sign of its last pulse). The target adds, for every non-empty subset
    of bits, the product of those bits' states - the k singles, C(k,2) pairs, ... and the full
    k-way parity. So the read-out grows as 2^k - 1 (3, 15, 63, 255 for k = 2, 4, 6, 8) instead of k,
    which is the point: the plain task's demand is linear in k and recruits ~k^0.16 active units;
    this one makes the demand combinatorial.

    Output channel ordering is by binary mask m = 1..2^k-1, where bit j of m says whether bit j is
    in the product: channel m-1 = prod_{j in m} bit_j. So channels 0..k-1 are NOT the singles;
    `self.subsets[c]` gives the tuple of bit indices for channel c. Before a bit's first pulse its
    state is 0, so every product containing it is 0 there too.

    n_outputs must equal 2**n_inputs - 1; the launcher passes both (Hydra cannot compute 2**k).
    """

    def __init__(self, n_steps, n_inputs, n_outputs, mu, n_flip_steps, batch_size=256, seed=None):
        """Same arguments as TaskNBitFlipFlop; n_outputs is checked against 2**n_inputs - 1."""
        if n_outputs != 2 ** n_inputs - 1:
            raise ValueError(f"n_outputs must be 2**n_inputs - 1 = {2 ** n_inputs - 1}, got {n_outputs}")
        TaskNBitFlipFlop.__init__(self, n_steps, n_inputs, n_outputs, mu, n_flip_steps, batch_size, seed)
        self.subsets = [tuple(j for j in range(n_inputs) if m >> j & 1) for m in range(1, 2 ** n_inputs)]

    def expand(self, bits):
        """Products of bit states over every non-empty subset.

        Args:
            bits: array (k, ...) of bit states in {-1, 0, +1}.
        Returns:
            array (2**k - 1, ...) - channel m-1 is the product over the bits in mask m. Built
            incrementally, P[m] = P[m without its top bit] * bit[top bit], so it costs one
            multiply per channel (255 at k=8) rather than one per (channel x member).
        """
        k = bits.shape[0]
        out = np.empty((2 ** k - 1,) + bits.shape[1:])
        for m in range(1, 2 ** k):
            top = m.bit_length() - 1
            rest = m ^ (1 << top)
            if rest == 0:
                out[m - 1] = bits[top]
            else:
                np.multiply(out[rest - 1], bits[top], out=out[m - 1])
        return out

    def generate_input_target_stream(self):
        """One trial; returns (input (k, T), target (2**k-1, T), condition) - see the parent."""
        inp, bits, cond = TaskNBitFlipFlop.generate_input_target_stream(self)
        return inp, self.expand(bits), cond

    def get_batch(self, shuffle=False):
        """A batch; returns (inputs (k, T, B), targets (2**k-1, T, B), conditions) - see the parent."""
        inp, bits, cond = TaskNBitFlipFlop.get_batch(self, shuffle=shuffle)
        return inp, self.expand(bits), cond


if __name__ == "__main__":
    # Self-check against an INDEPENDENT construction (itertools subsets + np.prod, not the mask
    # recursion), plus the per-channel target variance the R^2 comparison across k rests on.
    for k in (2, 3, 4, 6, 8):
        task = TaskNBitFlipFlopHyper(n_steps=300, n_inputs=k, n_outputs=2 ** k - 1, mu=4,
                                     n_flip_steps=10, batch_size=64, seed=0)
        parent = TaskNBitFlipFlop(n_steps=300, n_inputs=k, n_outputs=k, mu=4, n_flip_steps=10,
                                  batch_size=64, seed=0)
        inp, tgt, cond = task.get_batch()
        inp0, bits, cond0 = parent.get_batch()
        assert inp.shape == (k, 300, 64) and tgt.shape == (2 ** k - 1, 300, 64)
        assert np.array_equal(inp, inp0) and cond == cond0, "inputs must be the parent's"
        ref = {}
        for r in range(1, k + 1):
            for s in itertools.combinations(range(k), r):
                ref[s] = np.prod(bits[list(s)], axis=0)
        for c, s in enumerate(task.subsets):
            assert np.array_equal(tgt[c], ref[s]), f"channel {c} != product over {s}"
        assert set(task.subsets) == set(ref), "every non-empty subset exactly once"
        var = tgt.reshape(tgt.shape[0], -1).var(axis=1)
        by_size = {r: var[[len(s) == r for s in task.subsets]].mean() for r in range(1, k + 1)}
        print(f"k={k}: {2 ** k - 1} channels; target variance by subset size "
              + ", ".join(f"{r}:{v:.3f}" for r, v in by_size.items()))
    print("OK")
