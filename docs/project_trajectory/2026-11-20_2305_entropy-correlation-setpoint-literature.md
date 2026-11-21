## 2026-11-20 23:05 — Entropy and correlation set points: what the literature already has

**Why.** Pavel's question (via firstmate, 20 Nov): would a loss term that holds the entropy of each
unit's activity at a set point, or the correlation between units' activity at a set point, keep ReLU
units from becoming inactive during training? This entry lists the prior work the two terms build on and
says how each term differs from it. Rules for the test: `2026-11-20_2307_rules-entropy-correlation-setpoint.md`
written after this one. Branch `wp/entropy-setpoint`. Every reference below was checked on 20 Nov against
PubMed, arXiv, PMLR, a publisher's Crossref record or the publisher's page (an Internet Archive copy where
the publisher blocked automated access); the check is kept outside the repository at
`firstmate/data/entropy-setpoint/literature_check.md`. Claims taken from a paper's main text rather than
its abstract are marked "(main text)".

**Terms.**
- *Intrinsic plasticity*: a neuron changes its own excitability (gain or threshold) rather than its synapses.
- *Homeostatic plasticity*: a change that holds a neuron's activity near a set point.
- *Representation collapse*: in self-supervised learning, embeddings that become constant, or confined to a
  few dimensions (*dimensional collapse*).
- *Pre-activation*: in the standard network, the state h whose rate is relu(h).
- *Off on every step*: a ReLU unit whose h is below 0 at every step and trial of a batch. Any loss computed
  from its rate gives it exactly zero gradient.

### Intrinsic plasticity: the output entropy of one neuron

- Stemmler & Koch (1999), *How voltage-dependent conductances can adapt to maximize the information encoded
  by neuronal firing rate*, Nat. Neurosci. 2:521-527. An unsupervised rule adapts a model neuron's membrane
  conductances to maximise the information its firing rate carries about the stimulus; at a set mean rate,
  the neuron should fire at high rates only for rare events.
- Triesch (2005), *A gradient rule for the plasticity of a neuron's intrinsic excitability*, ICANN 2005,
  LNCS 3696:65-70. An information-theoretic gradient rule brings a neuron's firing-rate distribution "into an
  approximately exponential regime". The exponential is the maximum-entropy distribution of a non-negative
  rate at a fixed mean; the abstract does not state which parameters the rule adapts (the full text was not
  read).
- Triesch (2007), *Synergies between intrinsic and synaptic plasticity mechanisms*, Neural Comput.
  19:885-909. With synaptic plasticity, the same kind of intrinsic plasticity lets a neuron "discover
  heavy-tailed directions in the input".
- Lazar, Pipa & Triesch (2009), *SORN: a self-organizing recurrent neural network*, Front. Comput. Neurosci.
  3:23. A recurrent network of binary units with three local plasticity rules: spike-timing-dependent
  plasticity, synaptic normalisation, and intrinsic plasticity that moves each unit's threshold toward a
  target firing rate (main text, Eq. 7).
- Steil (2007), *Online reservoir adaptation by intrinsic plasticity for backpropagation-decorrelation and
  echo state learning*, Neural Netw. 20:353-364; Schrauwen, Wardermann, Verstraeten, Steil & Stroobandt
  (2008), *Improving reservoirs using intrinsic plasticity*, Neurocomputing 71:1159-1171. Intrinsic plasticity
  in reservoirs (fixed random recurrent weights, trained read-out): toward an exponential output distribution
  (Steil), and toward a Gaussian one for tanh units (Schrauwen et al.).

**How term (A) differs.** These rules adapt each neuron's own gain and threshold by a local rule, toward a
target distribution of its continuous output. Term (A) is a loss term, acting through gradient descent on all
weights; its quantity is the entropy of the unit's on/off state (how often it is above threshold), estimated
from the mean and s.d. of its pre-activation; its target is the median of active units in trained networks
(0.75 bits), not a maximum. The standard network of this project has its bias fixed at 0, so the
threshold route these rules use is closed: (A) can act only through the incoming weights.

### Firing-rate homeostasis

- Turrigiano, Leslie, Desai, Rutherford & Nelson (1998), *Activity-dependent scaling of quantal amplitude in
  neocortical neurons*, Nature 391:892-896. Blocking activity raised mEPSC (miniature excitatory postsynaptic
  current) amplitudes and blocking inhibition lowered them, all of a neuron's synapses in proportion, while
  firing rates returned close to control.
- Desai, Rutherford & Turrigiano (1999), *Plasticity in the intrinsic excitability of cortical pyramidal
  neurons*, Nat. Neurosci. 2:515-520. Two days without activity made cultured pyramidal neurons more sensitive
  to injected current, through their voltage-dependent conductances.
- Turrigiano & Nelson (2004), *Homeostatic plasticity in the developing nervous system*, Nat. Rev. Neurosci.
  5:97-107 (review).
- Zenke, Hennequin & Gerstner (2013), *Synaptic plasticity in neural networks needs homeostasis with a fast
  rate detector*, PLoS Comput. Biol. 9:e1003330; Zenke & Gerstner (2017), *Hebbian plasticity requires
  compensatory processes on multiple timescales*, Phil. Trans. R. Soc. B 372:20160259. In simulated recurrent
  networks with spike-timing-dependent plasticity, homeostasis must detect rate changes within seconds to
  minutes to keep activity stable; slow homeostasis alone does not.
- In this project: the rate penalty (`lambda_frm`, M09) is a per-unit rate set point applied by gradient. It
  reads rates, so a unit that is off on every step gets no gradient from it. `trainer.synaptic_scaling` (a
  participation set point, excitation up and inhibition down) was screened and set aside (Supplementary
  Table 3b); rescaling (M09) acts on inactive units directly.

### Decorrelation and variance floors against redundancy and collapse

- Barlow (1961), *Possible principles underlying the transformations of sensory messages*, in Rosenblith
  (ed.), *Sensory Communication*, MIT Press, 216-234 (pages from the MIT Press reissue record). Redundancy
  reduction as one aim of sensory relays.
- Földiák (1990), *Forming sparse representations by local anti-Hebbian learning*, Biol. Cybern. 64:165-170.
  Anti-Hebbian feedback between units learns a code with reduced dependency between its elements.
- Ecker, Berens, Keliris, Bethge, Logothetis & Tolias (2010), *Decorrelated neuronal firing in cortical
  microcircuits*, Science 327:584-587: in awake macaque V1, "even nearby neurons with similar orientation
  tuning show virtually no correlated variability". Renart, de la Rocha, Barthó, Hollender, Parga, Reyes &
  Harris (2010), *The asynchronous state in cortical circuits*, Science 327:587-590: recurrent networks reach
  arbitrarily low mean correlations despite shared input, because excitation and inhibition cancel it.
- Cogswell, Ahmed, Girshick, Zitnick & Batra (2016), *Reducing overfitting in deep networks by decorrelating
  representations*, ICLR (arXiv:1511.06068). DeCov minimises "the cross-covariance of hidden activations"; the
  loss is half the squared off-diagonal entries of the covariance matrix (main text, Eq. 2).
- Zbontar, Jing, Misra, LeCun & Deny (2021), *Barlow Twins: self-supervised learning via redundancy
  reduction*, ICML, PMLR 139:12310-12320 (arXiv:2103.03230). Drives the cross-correlation matrix between the
  embeddings of two distorted views of a sample toward the identity.
- Bardes, Ponce & LeCun (2022), *VICReg: variance-invariance-covariance regularization for self-supervised
  learning*, ICLR (arXiv:2105.04906). A term that keeps the variance of each embedding dimension above a
  threshold, and a term that decorrelates each pair of dimensions. In the main text (Section 4.1) the variance
  term is a hinge on each dimension's standard deviation over the batch (threshold γ = 1), the covariance term
  is the sum of squared off-diagonal covariances scaled by 1/d, and both act on the embeddings, the output of a
  network the paper calls the expander.
- Jing, Vincent, LeCun & Tian (2022), *Understanding dimensional collapse in contrastive self-supervised
  learning*, ICLR (arXiv:2110.09348). Embeddings can span only a low-dimensional subspace even under a
  contrastive loss.

**How term (B) differs.** DeCov, Barlow Twins and VICReg drive every off-diagonal covariance or correlation
toward 0. Term (B) holds each unit's *mean* correlation with the other units at a set point, a weaker
constraint (positive and negative pairs may cancel), on the pre-activations of a recurrent layer over time
and trials. VICReg's variance floor is the nearest thing in this group to a floor on activity, but it acts on
the expander's output, which has no ReLU; applied to ReLU rates, a hinge on the standard deviation would give
a unit that is off on every step zero gradient, because its rate and every derivative through the ReLU are 0.

### Dead ReLU units, and terms that reach them

- Maas, Hannun & Ng (2013), *Rectifier nonlinearities improve neural network acoustic models*, ICML Workshop on
  Deep Learning for Audio, Speech and Language Processing. Evaluates a variant of the rectifier "with a gradient
  more amenable to optimization". The abstract does not name it; the leaky rectifier (a small nonzero slope
  below 0, so no unit has exactly zero gradient) is usually attributed to this paper, which was not checked here.
- Ioffe & Szegedy (2015), *Batch normalization*, ICML, PMLR 37:448-456. Normalises each unit's input over the
  mini-batch. Before its learned scale and shift, every pre-activation has batch mean 0 and s.d. 1 (main text):
  a hard set point on the two moments term (A) reads. For a symmetric distribution of h, that puts a ReLU unit
  above threshold half the time.
- Lu, Shin, Su & Karniadakis (2020), *Dying ReLU and initialization: theory and numerical examples*, Commun.
  Comput. Phys. 28:1671-1706 (arXiv:1903.06733). A deep ReLU network "will eventually die in probability as
  the depth goes to infinite"; a randomised asymmetric initialisation prevents it.
- Sokar, Agarwal, Castro & Evci (2023), *The dormant neuron phenomenon in deep reinforcement learning*, ICML,
  PMLR 202:32145-32168 (arXiv:2302.12902). Deep RL networks accumulate inactive units; ReDo reinitialises a
  dormant unit's incoming weights and zeroes its outgoing weights (main text, Algorithm 1). Dohare et al. (2024),
  Nature 632:768-774, replace low-utility units, the route prune-and-duplicate takes in this project.
- Neftci, Mostafa & Zenke (2019), *Surrogate gradient learning in spiking neural networks: bringing the power
  of gradient-based optimization to spiking neural networks*, IEEE Signal Process. Mag. 36(6):51-63. Replaces
  the derivative of a non-differentiable activation in the backward pass by a smooth surrogate.
- Krueger & Memisevic (2016), *Regularizing RNNs by stabilizing activations*, ICLR (arXiv:1511.08400), penalise
  the squared distance between successive hidden-state norms; Merity, McCann & Socher (2017), *Revisiting
  activation regularization for language RNNs*, arXiv:1708.01009, use L2 and slowness penalties on RNN
  activations. Activation terms for RNNs, aimed at stability and regularisation, not at inactive units.

### The nearest prior terms (search for set-point losses against inactive units)

A search for a loss term with an entropy, firing-probability or correlation set point whose stated aim is to
keep ReLU units from dying found none with that stated aim. Three verified papers come closest:

| Paper | What it does | Difference from (A) and (B) |
|---|---|---|
| Cao, Ding, Lui & Huang (2018), *Improving GAN training via binarized representation entropy (BRE) regularization*, ICLR (arXiv:1805.03644) | On the pre-nonlinearity activity h of a GAN discriminator's rectifier layers, through softsign(h) = h/(\|h\| + ε) (Section 3.3): R_ME pushes each unit to be "activated half of the time" (Section 3.2); R_AC penalises the normalised overlap of binary activation patterns between pairs of samples, which the paper shows enforces pairwise independence of units (Section 3.1). Aim: better use of the discriminator's capacity; dead units are not mentioned in the sections read | The closest method to (A): it also reads pre-activations, so it reaches a unit that is off on every step. Its set point is half on (1 bit), a single maximum; (A)'s is 0.75 bits from trained networks, two-sided in entropy, with p estimated from a Gaussian over the batch. R_AC works over pairs of samples; (B) works over pairs of units. Feedforward discriminator, not an RNN |
| Peer, Keulen, Stabinger, Piater & Rodríguez-Sánchez (2022), *Improving the trainability of deep neural networks through layerwise batch-entropy regularization*, TMLR (arXiv:2208.01134) | A loss that holds each layer's batch-entropy at a per-layer set point; plain fully connected and convolutional networks of 500 layers then train without skip connections or normalisation | Per layer, not per unit; aims at the flow of information through depth, not at inactive units |
| Zenke & Vogels (2021), *The remarkable robustness of surrogate gradient learning for instilling complex function in spiking neural networks*, Neural Comput. 33:899-925 | A loss with a per-neuron lower bound on spike counts, "to prevent quiescent units in the hidden layers" (bioRxiv preprint, Methods) | A per-unit lower bound on activity in spiking networks trained with surrogate gradients; on our reading, the gradient reaches a silent neuron through the surrogate derivative, not through a statistic of its input |

### Where the literature leaves the two terms

- **(A)** combines three known pieces: a set point on how often a unit is above threshold (BRE's R_ME; batch
  normalisation as a hard version), intrinsic plasticity's view that a unit's output entropy should be held at
  a target, and a pre-activation statistic that keeps a gradient on a unit that never fires. No paper found
  applies such a term to the inactive units of a task-trained RNN or measures active-unit counts with it.
- **(B)** is a per-unit, mean-correlation version of the decorrelation terms of DeCov, Barlow Twins and VICReg,
  applied to the pre-activations of a recurrent layer. None of those papers is about inactive units, and a
  Pearson correlation is unchanged by shifting a unit's h, so (B) cannot raise an inactive unit's mean drive
  directly (rules entry, choice 5).
- **The searches cannot prove that no such work exists.** None of the papers above counts active units in a
  gradient-trained RNN, which is the measurement of this experiment.
