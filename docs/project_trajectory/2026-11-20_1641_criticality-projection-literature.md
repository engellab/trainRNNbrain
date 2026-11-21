## 2026-11-20 16:41 — Criticality and W_rec projection: what the literature already has

**Why.** Pavel's idea (via firstmate, 20 Nov): initialising the recurrent weights well lets gradients
propagate through time; training may degrade that property; so project W_rec back toward where it
started during training, as a kind of homeostatic plasticity, and see whether fewer units go inactive.
This entry lists the prior work the test builds on and says how each of the four rules tested here
differs from it. Rules for the test: `2026-11-20_1643_rules-criticality-projection.md`. Branch
`wp/criticality-proj`. Every reference below was checked against arXiv, PMLR, the conference page or
the publisher record on 20 Nov; the check is kept outside the repository at
`firstmate/data/criticality-proj/literature_check.md`.

**Terms.** *Edge of chaos*: the boundary between dynamics in which small perturbations die out
(ordered) and dynamics in which they grow (chaotic). *Largest Lyapunov exponent* (λ1): the mean
exponential growth rate per step of a small perturbation along a trajectory; λ1 = 0 at the edge,
negative in the ordered regime. *Dynamical isometry*: every singular value of the input-output
Jacobian is close to 1, so gradients neither grow nor shrink with depth. *Spectral radius* ρ(W): the
largest eigenvalue modulus. *σ1(W)*: the largest singular value.

### Signal propagation at initialisation

- Sompolinsky, Crisanti & Sommers (1988), *Chaos in random neural networks*, Phys. Rev. Lett. 61:259.
  In a random network with asymmetric couplings the zero fixed point is stable for gain g < 1 and the
  dynamics are chaotic for g > 1.
- Poole, Lahiri, Raghu, Sohl-Dickstein & Ganguli (2016), *Exponential expressivity in deep neural
  networks through transient chaos*, NeurIPS 29 (arXiv:1606.05340). Random deep networks have an
  order-to-chaos transition set by the weight and bias variances.
- Schoenholz, Gilmer, Ganguli & Sohl-Dickstein (2017), *Deep information propagation*, ICLR
  (arXiv:1611.01232). The depth over which a signal survives diverges at the order-chaos boundary, so
  very deep networks train only when initialised near it.
- Pennington, Schoenholz & Ganguli (2017), *Resurrecting the sigmoid in deep learning through
  dynamical isometry: theory and practice*, NeurIPS 30 (arXiv:1711.04735). Their abstract states that
  "ReLU networks are incapable of dynamical isometry". The result is for feedforward layers with
  independent weights, not for an RNN that reuses one W_rec at every step.
- Chen, Pennington & Schoenholz (2018), *Dynamical isometry and a mean field theory of RNNs: gating
  enables signal propagation in recurrent neural networks*, ICML, PMLR 80:873-882 (arXiv:1806.05394).
  A gated cell trains from a much wider range of initialisations than a vanilla RNN.
- Saxe, McClelland & Ganguli (2014), *Exact solutions to the nonlinear dynamics of learning in deep
  linear neural networks*, ICLR (arXiv:1312.6120). Random orthogonal initialisation gives learning
  times that do not grow with depth.
- Le, Jaitly & Hinton (2015), *A simple way to initialize recurrent networks of rectified linear
  units*, arXiv:1504.00941 (preprint, no peer-reviewed venue found). A ReLU RNN started from the
  identity recurrent matrix (IRNN) matched LSTMs on long-range toy tasks.

All of this work is about initialisation. None of it measures whether the property survives training,
which is the claim tested first here.

**What it implies for the networks of this project** (derived here, not taken from the papers). One
step of the standard network is h_{t+1} = h_t + α(−h_t + W_rec relu(h_t) + W_inp u_t), α = 0.1, so its
Jacobian is J_t = (1 − α)I + α W_rec D_t with D_t = diag(1[h_t > 0]). An inactive unit zeroes its column
of W_rec D_t: a perturbation of its h reaches no other unit. With every unit inactive J_t = (1 − α)I and
λ1 = log 0.9 = −0.105 per step. For Gaussian W with spectral radius r and a fraction f of units above
threshold, ρ(W D) is about r√f; at the initial r = 1.2 it falls below 1 once f < 0.69. So inactive units
lower the effective gain, and the dynamics can move toward the ordered regime even when ρ(W_rec) grows.

### Methods that already constrain the recurrent matrix or its exponents

| Method | What it does | Closest rule here and the difference |
|---|---|---|
| Engelken (2023), *Gradient flossing: improving gradient descent through dynamic control of Jacobians*, NeurIPS 36 (arXiv:2312.17306) | Adds Σ_{i≤k} λ_i² to the loss for the k largest exponents, computed by pushing k tangent vectors through the one-step Jacobians with QR re-orthonormalisation; gradients pass through the QR step. Applied before training or in short episodes during it; the paper warns that flossing throughout training can lower task performance. Mostly tanh RNNs | **(d)**. Same target, λ1 → 0. Rule (d) is not a loss term: it measures λ1 only and rescales W_rec by one scalar, so it cannot change the shape of W_rec, and it runs through all of training |
| Miyato, Kataoka, Koyama & Yoshida (2018), *Spectral normalization for generative adversarial networks*, ICLR (arXiv:1802.05957) | Divides each weight matrix by σ1, estimated by power iteration, at every step | **(a)**. Rule (a) restores ρ (not σ1; for Gaussian W, σ1 ≈ 2ρ) to its initial value (not 1), by a periodic projection |
| Helfrich & Ye (2020), *Eigenvalue normalized recurrent neural networks for short term memory*, AAAI 34:4115 (arXiv:1911.07964) | W_rec = T / (ρ(T) + ε), trained by gradient descent | **(a)** as a reparametrisation, with target just below 1 |
| Lewandowski, Bortkiewicz, Kumar, György, Schuurmans, Ostaszewski & Machado (2025), *Learning continually by spectral regularization*, ICLR (arXiv:2406.06811) | Penalty Σ_l (σ1(W_l)² − 1)² to keep plasticity in continual learning; feedforward and RL networks | **(a)** as a soft penalty on σ1, target 1 |
| Vorontsov, Trabelsi, Kadoury & Pal (2017), *On orthogonality and learning recurrent networks with long term dependencies*, ICML, PMLR 70:3570 (arXiv:1702.00071); Zhang, Lei & Dhillon (2018), *Stabilizing gradients for deep neural networks via efficient SVD parameterization*, ICML, PMLR 80:5806 | W_rec = U S V^T with the singular values confined to a margin around 1 (Vorontsov) or explicitly controlled (Zhang) | **(b)**. Rule (b) keeps the learned U and V and puts S back to the initial Gaussian spectrum (about 0 to 2.4 at ρ = 1.2), not to a band around 1 |
| Arjovsky, Shah & Bengio (2016), *Unitary evolution recurrent neural networks*, ICML, PMLR 48:1120; Lezcano-Casado & Martínez-Rubio (2019), *Cheap orthogonal constraints in neural networks*, ICML, PMLR 97:3794 | Recurrent matrix kept unitary or orthogonal | **(b)** with S fixed to the identity |
| Kerg et al. (2019), *Non-normal recurrent neural network (nnRNN)*, NeurIPS 32 (arXiv:1905.12080) | Schur parametrisation keeps every eigenvalue at modulus 1 | The eigenvalue counterpart of (b) |
| Sedghi, Gupta & Long (2019), *The singular values of convolutional layers*, ICLR (arXiv:1805.10408) | Clips each convolutional layer's singular values every 100 steps | A periodic projection like (b), on the top of the spectrum only |
| Salimans & Kingma (2016), *Weight normalization*, NeurIPS 29 (arXiv:1602.07868) | Each unit's weights written as w = g v / ‖v‖ | **(c)** with g frozen at the initial row norm, applied as a reparametrisation every step |
| Lyle, Zheng, Khetarpal, Martens, van Hasselt, Pascanu & Dabney (2024), *Normalization and effective learning rates in reinforcement learning*, NeurIPS (arXiv:2407.01800) | Normalize-and-Project: rescales each layer's weights to their initial norm; projecting every step and every 1,000 steps gave similar results | **(c)** per layer instead of per unit; no RNNs |
| Kumar, Marklund & Van Roy (2024), *Maintaining plasticity in continual learning via regenerative regularization*, CoLLAs, PMLR 274:410 (arXiv:2308.11958) | L2 penalty on the distance from the initial weights | A pull of every weight toward its initial value; (a)-(c) restore only a scale, a spectrum or a row norm |

### Homeostatic scaling

- Turrigiano, Leslie, Desai, Rutherford & Nelson (1998), *Activity-dependent scaling of quantal
  amplitude in neocortical neurons*, Nature 391:892-896. Blocking activity raised mEPSC (miniature
  excitatory postsynaptic current) amplitudes and blocking inhibition lowered them, each synapse in
  proportion to its strength. The biological rule responds to the neuron's firing rate.
- Schubert & Gros (2021), *Local homeostatic regulation of the spectral radius of echo-state networks*,
  Front. Comput. Neurosci. 15:587721. Local scaling rules drive an untrained reservoir's spectral
  radius to a target. No gradient training.
- In this project, `trainer.synaptic_scaling` (a participation set-point, excitatory weights up and
  inhibitory weights down, row norm preserved) was screened at N = 1,000 for 40,000 iterations and not
  continued (paper_audit C6, Supplementary Table 3b). Rule (c) is a different rule: it restores each
  unit's incoming norm regardless of the unit's activity, as weight normalisation does, so it does
  nothing for an inactive unit whose norm has not changed.

### Where the literature leaves the idea

- **Rule (a)** is spectral normalisation or eigenvalue normalisation with the initial ρ as the target.
  It is not new, and it is tested here as a variant of those methods.
- **Rule (b)** is an SVD-constrained RNN whose spectrum is frozen at the initial Gaussian one instead of
  at 1.
- **Rule (c)** is weight normalisation with frozen gains.
- **Rule (d)** is the closest to new. No paper found rescales W_rec by a non-gradient scalar to drive
  λ1 toward 0 during gradient training; gradient flossing is the nearest method.
- **The searches cannot prove that no such work exists.** None of the papers above measures the number
  of active units, so none tests the claim of this experiment: that restoring the initial regime keeps
  units active.
- **Dohare et al. (2024)**, *Loss of plasticity in deep continual learning*, Nature 632:768-774,
  replaces low-utility units, the route prune-and-duplicate takes in this project.
- **Lyle et al. (2023)**, *Understanding plasticity in neural networks*, ICML, PMLR 202:23190
  (arXiv:2303.01486), report that loss of plasticity tracks changes in loss-landscape curvature and
  often occurs without saturated units.
