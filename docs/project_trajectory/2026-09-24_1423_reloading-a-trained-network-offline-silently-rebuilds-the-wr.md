## 2026-09-24, 14:23 — Reloading a trained network offline silently rebuilds the wrong architecture

Measuring activity dimensionality needs the trained weights run forward offline. Three attempts
produced three different answers for the same networks, and every one of them looked plausible.
The only reason the errors were caught is a gate that recomputes each network's r2 and refuses to
report anything derived from a forward pass that does not reproduce the stored value.

| attempt | recomputed r2 (stored 0.944) |
|---|---|
| hand-copy of W_rec, W_inp, W_out | 0.78 |
| `RNN_torch.set_params(checkpoint)` | **−0.08** |
| architecture from the config, weights from the checkpoint | 0.78 |

### The trap

**`LastParams_*.npz` does not store the architecture flags.** Its keys are `N`, `W_inp`, `W_out`,
`W_rec`, `activation_args`, `bias`, `bias_range`, `dt`, `gamma`, `input_mask`, `output_mask`,
`recurrent_mask`, `tau`, `weight_boundary_eps`, `y_init`. There is no `equation_type`, no `dale`,
no `io_nonnegativity` and no `self_connections`.

`set_params` handles the absence by falling back to the legacy configuration — its own comment says
"legacy nets predate these flags: absent -> the old behaviour (both constraints on)". That is right
for old networks and wrong for every network trained with `rnn_relu_standard`, which sets
`io_nonnegativity: false` and `self_connections: true`. The rebuilt network therefore applies
`|W_inp|` in its forward pass and zeroes its recurrent diagonal. Nothing errors: the weights load,
the forward pass runs, the activity looks reasonable, and only r2 reveals that it is a different
network.

**Rule: build the architecture from the config the launcher used, and take only the learned
quantities from the checkpoint.** Never rely on `set_params` alone for these nets.

### Still unresolved

With the architecture right, recomputed r2 is 0.78 against a stored 0.944, consistent across
networks, while the batch-to-batch spread on one network is only ±0.005 — so batch randomness is
not the explanation either. The remaining suspect is which checkpoint the folder name describes:
each directory holds both `BestParams` and `LastParams`, and the stored score comes from
`trainer.eval_step(..., noise=True)` on whatever the trainer held at the end. A test of Best
against Last is running.

⚠️ **No dimensionality number has been reported from any of these attempts.** An early table showing
`dead` dropout at 30 activity dimensions against a control's 6 was computed from the first, wrong
rebuild and is withdrawn. It was entirely plausible and entirely fabricated by a broken forward
pass, which is the argument for keeping the gate rather than widening it when it fails.

---

