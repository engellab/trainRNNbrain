import datajoint as dj
dj.config['database.host'] = 'datajoint-tengel.pni.princeton.edu'
dj.config['database.user'] = 'pt1290'
dj.config['database.password'] = 'a9Ab?spCKz$Zh@24h'
dj.config['display.limit'] = 100
dj.config["enable_python_native_blobs"] = True
dj.conn(reset=True)
schema = dj.schema('ptolmachev_RNNs')


@schema
class TaskDJ(dj.Manual):
    definition = """
       task_name: char(16)                                           
       ---
       n_steps: smallint unsigned
       n_inputs: tinyint unsigned
       n_outputs: tinyint unsigned
       task_params : blob
       mask : blob
    """

@schema
class TrainerDJ(dj.Manual):
    definition = """
       -> TaskDJ 
       trainer_id: int 
       ---
       max_iter: int        
       tol: float
       lr: float
       lambda_orth: float
       lambda_r: float
       same_batch : tinyint
       shuffle : tinyint
    """

@schema
class RNNDJ(dj.Manual):
    definition = """
       -> TaskDJ
       -> TrainerDJ
       rnn_timestamp: char(14)                                      # unique model id
       ---
       n: int                                                       # number of nodes in the RNN
       activation_name: enum('relu', 'tanh', 'sigmoid', 'softplus') # name of the activation function used in the dynamics
       constrained: tinyint                                         # boolean variable, either True or False, using biologically plausible connectivity or not
       dt: float                                                    # Euler integration timestep
       tau: float                                                   # Dynamical time-scale
       sr: float                                                    # spectral radius of the recurrent conenctivity
       connectivity_density_rec: Decimal(1,0)                       # opposite of sparsity of the connectivity. 1 - fully connected network
       sigma_rec: float
       sigma_inp: float
       w_inp : blob
       w_rec : blob
       w_out: blob
       b_rec: blob
    """

@schema
class CDDMRNNAnalysisDJ(dj.Manual):
    definition = """
       -> TaskDJ
       -> RNNDJ                                          
       ---
       mse_score: float
       fp_data: mediumblob  
       psycho_data: mediumblob
       la_data: mediumblob  
    """