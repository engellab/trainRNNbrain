## Training continuous time Recurrent Neural Networks (RNNs) on various behavioral tasks

This project aims to establish a pipeline for seamlessly defining the behavioral task and training RNNs on it using
backpropagation (BPTT) on *PyTorch*.

It also implements a useful class for the post-training task-performance analysis, as well as a class for analysis of
the RNN dynamics: computing its dynamics fixed-points structure for a given input.

### Some examples:

<p align="center">
<img src="https://github.com/engellab/RNN_training_pipeline/blob/main/img/fixed%20points%203BitFlipFlop%20task.gif?raw=true"/>
</p>

<center>

*Fixed point structure revealed after training an RNN to perform a 3 bit flip-flop task*

</center>

__________________________________
<p align="center">
<img src="https://github.com/engellab/RNN_training_pipeline/blob/main/img/random_trials_MemoryAnti_task.png" width="500">
</p>

<center>

*Random trials after training the RNN on 2 bit flip-flop task*

</center>

__________________________________

<p align="center">
<img src="https://github.com/engellab/RNN_training_pipeline/blob/main/img/FixedPoints_MemoryAntiNumber.png" width="500">
</p>

<center>

*Fixed point structure for MemoryAntiNumber task:
The first line attractor (blue-red points, appearing for the input during the stimulus-presentation stage) lies in the
nullspace of the W_out. The second line-attractor (violet-tomato points, appearing for the input provided on the recall
stage) has some projection on the output axes*
</center>

__________________________________


<p align="center">
<img src="https://github.com/engellab/RNN_training_pipeline/blob/main/img/fixed%20points%20MemoryAnti%20task.gif?raw=true"/>
</p>

<center>

*Fixed point structure in the MemoryAntiAngle task:
same as for the line attractors in MemoryAntiNumber task, but instead of the line attractors, the networks forms ring
attractors.*
</center>

__________________________________

### Continuous-time RNN description

The dynamics for RNN are captured by the following equations:

<img src="https://latex.codecogs.com/svg.image?\begin{align*}\tau&space;\mathbf{\dot{x}}&space;&=&space;-\mathbf{x}&space;&plus;&space;[W_{rec}\mathbf{x}&space;&plus;&space;W_{inp}&space;(\mathbf{u}&space;&plus;&space;\xi_{inp})&space;&plus;&space;\mathbf{b}_{rec}&space;&plus;&space;\xi_{rec}]_&plus;&space;\\\text{out}&space;&=&space;W_{out}&space;\mathbf{x}&space;\end{align*}&space;" title="https://latex.codecogs.com/svg.image?\begin{align*}\tau \mathbf{\dot{x}} &= -\mathbf{x} + [W_{rec}\mathbf{x} + W_{inp} (\mathbf{u} + \xi_{inp}) + \mathbf{b}_{rec} + \xi_{rec}]_+ \\\text{out} &= W_{out} \mathbf{x} \end{align*} " />

Where "\tau" is the time constant, "x" is the state vector of the RNN, "u" is an input vector, "W rec" is the recurrent
connectivity of the RNN, "W inp" - matrix of input connectivities distributing input vector "u" to the neural nodes, "b
rec" is a bias in the recurrent connectivity, "\xi" is some gaussian random noise. The output of the network is provided
by the readout matrix "W out" applied to the neural nodes.

There are two classes implementing RNN dynamics:

- **RNN_pytorch** -- used for training the network on the task
- **RNN_numpy** -- used for performance analysis, fixed point analysis, and easy plotting.

### Task definition

Each task has its own class specifying the structure of (input, target output) of the behavior. It should contain two
main methods:

- `generate_input_target_stream(**kwargs)` -- generates a single (input, target output, condition) tuple with specified
  parameters
- `get_batch(**kwargs)` -- generates a batch of inputs, targets and conditions. The batch dimension is the last.

The implemented example tasks are:

- Context-Dependent Decision Making
- Delayed Match to Sample
- 3 Bit Flip-Flop
- MemoryAntiNumber
- MemoryAntiAngle

Descriptions of these tasks are provided in the comments in the relevant task classes.

One can easily define their own task following the provided general template.

### Training

During the training, the connectivity matrices W_rec, W_inp, W_out are iteratively modified to minimize a loss function:
the lower the loss function, the better the network performs the task.

The training loop is implemented in the **Trainer** class, which accepts the task and the RNN_pytorch instance. Trainer
implements three main methods:

- `train_step(input_batch, target_batch)` -- returns the loss-value for a given batch, (linked to the computational
  graph to compute the gradient w.r.t connectivity weights) as well as the vector of losses on each individual trial
- `eval_step(input_batch, target_batch)` -- returns the loss value for a given batch, detached from the gradient.
- `run_training(**kwargs)` -- implements an iterative update of connectivity parameters, minimizing a loss function

### Performance Analysis

The class **PerformanceAnalyzer** accepts the RNN_numpy instance and a Task instance and has two main methods:

- `get_validation_score(scoring_function, input_batch, target_batch, **kwargs)` -- runs the network with the specified
  inputs and calculates the mean loss between the predicted and target outputs using the specified scoring function.
- `plot_trials(input_batch, target_batch, **kwargs)` -- generates a figure plotting multiple predicted outputs as a
  response to specified inputs, as well as shows target outputs for comparison.

One can extend the base class by defining task-specific PerformanceAnalyzer
(see AnalyzerCDDM as an example)

### Fixed-point Analysis

The fixed-point analysis is implemented in the **DynamicSystemAnalyzer** class and accepts RNN_numpy instance.

The class contains three methods:

- `get_fixed_points(Input_vector, mode, **kwargs)`  -- calculates stable and unstable fixed points of the RNN's dynamics
  for a given input. It searches for exact fixed points if *mode = 'exact'* option, using `scipy.fsolve` methods applied
  to the right-hand side of the dynamics equations. Alternatively, if *mode = 'approx'*  it searches for 'slow points'
  -- points where RHS of dynamics is approximately zero. In the latter case, the cut-off threshold for a point is
  controlled by the parameter *'fun_tol'*.
- `plot_fixed_points(projection, P)` -- assumes that the fixed points has been calculated with `get_fixed_points` method
  for maximum three input vectors, If th projection matrix `P' is not specified, assembles the fixed points into an
  array, performs the PCA on them and projects the points on either first 2 (projection='2D') or first 3 (
  projection='3D') PCs, returning the figure with the projected fixed points.
- `compute_point_analytics(point, Input_vector, **kwargs)` -- at a given point in the state-space, and given input to
  the RNN, calculate statistics of the point:
  value of the |RHS|^2, Jacobian, eigenvalues, and the principle left and right eigenvectors.

### Saving the data

When initialized, DataSaver creates a dedicated data_folder and stores its address as a 'data_folder' parameter. It has
two methods:

- `save_data(data, filename)` -- saves either a pickle or JSON file containing the data into the 'data_folder'
- `save_figure(figure, filename)` -- saves a figure as a png-file into the 'data_folder'

Integration with DataJoint is coming
