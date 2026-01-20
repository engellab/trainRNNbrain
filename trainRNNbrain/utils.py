import textwrap
import numpy as np
from pathlib import Path
import torch
import ast
import inspect, importlib.util, pathlib
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def in_the_list(x, x_list, diff_cutoff=1e-6):
    ''' Check if x is in x_list within a certain L2 norm cutoff. '''
    for i in range(len(x_list)):
        diff = np.linalg.norm(x - x_list[i], 2)
        if diff < diff_cutoff:
            return True
    return False


def count_return_values(func):
    ''' Count the number of return values of a function. '''
    try:
        source = inspect.getsource(func)
        source = textwrap.dedent(source)  # <-- fix
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Return):
                if isinstance(node.value, ast.Tuple):
                    return len(node.value.elts)
                elif node.value is not None:
                    return 1
                else:
                    return 0
        return None
    except Exception as e:
        print(f"Error in count_return_values: {e}")
        return None
    

# DATA ANALYSIS UTILITIES
def composite_lexicographic_sort(matrix1, matrix2, dale_mask):
    """
    Sorts rows by:
    1. Lexicographic order of matrix1
    2. Then matrix2
    3. Then dale_mask (-1 before +1)
    Args:
        matrix1 (np.ndarray): shape (N, D1), one-hot rows
        matrix2 (np.ndarray): shape (N, D2), one-hot rows
        dale_mask (np.ndarray): shape (N,), values in {-1, +1}
    Returns:
        perm (np.ndarray): permutation of row indices
    """
    assert matrix1.shape[0] == matrix2.shape[0] == dale_mask.shape[0], "Mismatch in row count"
    idx1 = np.argmax(matrix1, axis=1)  # primary key
    idx2 = np.argmax(matrix2, axis=1)  # secondary key
    sort_keys = np.lexsort((dale_mask, idx2, idx1))
    return sort_keys

def compute_intercluster_weights(W_inp, W_rec, W_out, labels):
    """
    Compute average inter-cluster connectivity matrices.

    Args:
        W_inp: (N, I)
        W_rec: (N, N)
        W_out: (O, N)
        labels: (N,) array of integers in [0, C-1]

    Returns:
        w_inp: (C, I)
        w_rec: (C, C)
        w_out: (O, C)
    """
    N = labels.shape[0]
    C = np.max(labels) + 1  # number of clusters

    w_inp = np.zeros((C, W_inp.shape[1]))
    w_rec = np.zeros((C, C))
    w_out = np.zeros((W_out.shape[0], C))

    for i in range(C):
        idx_i = np.where(labels == i)[0]
        if len(idx_i) == 0:
            continue
        # Average input into cluster i
        w_inp[i] = W_inp[idx_i].mean(axis=0)

        for j in range(C):
            idx_j = np.where(labels == j)[0]
            if len(idx_j) == 0:
                continue
            # Average recurrent weight from cluster j to i
            submatrix = W_rec[np.ix_(idx_i, idx_j)]
            w_rec[i, j] = submatrix.mean()

    # Average output from each cluster
    for j in range(C):
        idx_j = np.where(labels == j)[0]
        if len(idx_j) == 0:
            continue
        w_out[:, j] = W_out[:, idx_j].mean(axis=1)

    return w_inp, w_rec, w_out

def permute_matrices(W_inp, W_rec, W_out, perm):
    W_inp_ = W_inp[perm, :]
    W_rec_ = W_rec[perm, :]
    W_rec_ = W_rec_[:, perm]
    W_out_ = W_out[:, perm]
    return W_inp_, W_rec_, W_out_

def cluster_neurons(trajectories, dale_mask=None, n_clusters=(8, 4)):
    if dale_mask is None:
        F = trajectories.reshape(trajectories.shape[0], -1)
        pca = PCA(n_components=10)
        pca.fit_transform(F)
        P = pca.components_.T
        D = F @ P
        cl = KMeans(n_clusters=n_clusters)
        cl.fit(D)
        labels = cl.labels_
    else:
        idx_pos = np.where(dale_mask > 0)[0]
        idx_neg = np.where(dale_mask < 0)[0]
        trajectories_pos = trajectories[idx_pos, ...]
        trajectories_neg = trajectories[idx_neg, ...]
        labels_pos = cluster_neurons(trajectories_pos, dale_mask=None, n_clusters=n_clusters[0])
        labels_neg = cluster_neurons(trajectories_neg, dale_mask=None, n_clusters=n_clusters[1])

        labels = np.zeros_like(dale_mask)
        labels_neg = len(np.unique(labels_pos)) + np.array(labels_neg)
        labels[np.where(dale_mask==1)[0]] = labels_pos
        labels[np.where(dale_mask == -1)[0]] = labels_neg
    return labels

# MATH UTILITIES

def orthonormalize(W):
    for i in range(W.shape[-1]):
        for j in range(i):
            W[:, i] = W[:, i] - W[:, j] * np.dot(W[:, i], W[:, j])
        W[:, i] = W[:, i] / np.linalg.norm(W[:, i])
    return W

def sort_eigs(E, R):
    # sort eigenvectors
    data = np.hstack([E.reshape(-1, 1), R.T])
    data = np.array(sorted(data, key=lambda l: np.real(l[0])))[::-1, :]
    E = data[:, 0]
    R = data[:, 1:].T
    return E, R


def make_orientation_consistent(vectors, num_iter=10):
    vectors = np.stack(np.stack(vectors))
    for i in range(num_iter):  # np.min(dot_prod) < 0:
        average_vect = np.mean(vectors, axis=0)
        average_vect /= np.linalg.norm(average_vect)
        dot_prod = vectors @ average_vect
        vectors[np.where(dot_prod < 0)[0], :] *= -1
    return vectors

def cosine_sim(A, B):
    v1 = A.flatten() / np.linalg.norm(A.flatten())
    v2 = B.flatten() / np.linalg.norm(B.flatten())
    return np.round(np.dot(v1, v2), 3)


# FILE IMPORTING UTILITIES
def import_any(target):
    ''' Import a module or class from a dotted path or file path. '''
    # file path
    p = pathlib.Path(str(target))
    if p.suffix == ".py" and p.exists():                       # file path
        spec = importlib.util.spec_from_file_location(p.stem, str(p))
        mod  = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        return mod
    # dotted path: module or module.Class
    try:
        return importlib.import_module(target)                 # module
    except Exception:
        mod_name, attr = target.rsplit(".", 1)                 # module.Class
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr)
    
def get_source_code(obj_or_mod):
    """Return source code for a module, class, function, or instance."""
    if inspect.ismodule(obj_or_mod):
        target = obj_or_mod
    elif inspect.isclass(obj_or_mod) or inspect.isfunction(obj_or_mod) or inspect.ismethod(obj_or_mod):
        target = obj_or_mod
    else:
        target = obj_or_mod.__class__ if hasattr(obj_or_mod, "__class__") else inspect.getmodule(obj_or_mod)
    try:
        return inspect.getsource(target)
    except OSError:
        # fallback if loaded from .pyc
        import importlib.util, pathlib
        path = pathlib.Path(getattr(target, "__file__", ""))
        if path.suffix == ".pyc":
            path = pathlib.Path(importlib.util.source_from_cache(str(path)))
        return path.read_text()
    
def filter_kwargs(callable_obj, params: dict):
    ''' Filter parameters to those accepted by callable_obj and return an OmegaConf DictConfig. '''
    sig = inspect.signature(callable_obj)
    flags = {"allow_objects": True}
    cfg = params if isinstance(params, DictConfig) else OmegaConf.create(params or {}, flags=flags)
    # if it accepts **kwargs, pass everything through
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return cfg
    allowed = {name for name, p in sig.parameters.items()
               if name != 'self' and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    hydra_keys = [k for k in cfg.keys() if str(k).startswith("_")]
    filtered_keys = hydra_keys + [k for k in cfg.keys() if k in allowed]
    filtered_dict = {k: cfg[k] for k in filtered_keys}
    return OmegaConf.create(filtered_dict, flags=flags)
    
def jsonify(x):
    if hasattr(x, "items") and not isinstance(x, (str, bytes)):
        return {jsonify(k): jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonify(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonify(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    if torch is not None and isinstance(x, torch.Tensor):
        return x.item() if x.numel() == 1 else jsonify(x.detach().cpu().tolist())
    return x


def unjsonify(dct):
    dct_unjsonified = {}
    for key in dct:
        val = dct[key]
        if isinstance(val, dict):
            dct_unjsonified[key] = unjsonify(val)
        elif isinstance(val, list):
            try:
                dct_unjsonified[key] = np.array(val)
            except:
                # Leave as list if it can't be safely converted
                dct_unjsonified[key] = val
        elif torch.is_tensor(val):
            dct_unjsonified[key] = val.detach().cpu().numpy()
        else:
            dct_unjsonified[key] = val
    return dct_unjsonified

def get_project_root():
    return Path(__file__).parent.parent

def numpify(function_torch):
    return lambda x: function_torch(torch.Tensor(x)).detach().numpy()

def make_subfolder_tag(cfg, net_params, score, taskname):
    ''' Create a concise tag string summarizing the training configuration. '''
    t = cfg.trainer
    def abbr(k):
        return 'L'+k[7:] if k.startswith('lambda_') else ''.join(w[0].upper() for w in k.split('_') if w)
    def fmt(v):
        return f"{v:.3g}" if isinstance(v, float) else str(int(v)) if isinstance(v, bool) else str(v)

    td = {k: getattr(t, k) for k in dir(t) if not k.startswith('_') and not callable(getattr(t, k))}
    lambdas = {k: v for k, v in td.items() if k.startswith('lambda_') and v is not None}
    core_keys = [k for k in ('learning_rate','lr','max_iter','dropout','drop_rate','weight_decay','orth_input_only') if k in td]

    parts = [
        f'{score}_{taskname}_{net_params["activation_args"]["name"]}',
        f'N={net_params["N"]}',
        *[f'{abbr(k)}={fmt(td[k])}' for k in core_keys],
        *[f'{abbr(k)}={fmt(lambdas[k])}' for k in sorted(lambdas)]
    ]
    return ';'.join(parts)
