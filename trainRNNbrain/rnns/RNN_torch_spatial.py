from copy import deepcopy
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

def visualize_connectivity_3d(W_rec, W_inp, W_out, recurrent_mask, dale_mask, output_mask, input_mask,
                              pos, d=1.0, max_edges=4000, point_size=18, line_alpha=0.08):
    Nf = torch.tensor(float(recurrent_mask.shape[0]))
    side = torch.sqrt(Nf / 100.0).item()  # <- same neuronal_density as above
    to_np = lambda t: (t.detach().cpu().numpy() if hasattr(t,"detach") else np.asarray(t))
    P = to_np(pos); N = P.shape[0]; z,x,y = P[:,0],P[:,1],P[:,2]
    R = to_np(recurrent_mask).astype(bool)
    D = to_np(dale_mask).reshape(-1) if to_np(dale_mask).ndim==1 and to_np(dale_mask).shape[0]==N else \
        np.sign(to_np(W_rec).sum(0)); D[D==0]=1.0
    cols = np.where(D>0,'r','b')

    ii,jj = np.where(R); keep = ii<jj; ii,jj = ii[keep],jj[keep]
    if ii.size>max_edges:
        sel = np.random.choice(ii.size, max_edges, replace=False); ii,jj = ii[sel],jj[sel]

    fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
    for a in (ax.xaxis,ax.yaxis,ax.zaxis): a.set_pane_color((1,1,1,0))
    ax.set_xlim(0,side); ax.set_ylim(0,side); ax.set_zlim(0,d)
    ax.set_xticks([0,side]); ax.set_yticks([0,side]); ax.set_zticks([0,d])
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])

    def slab(z0,z1,color,alpha=0.15):
        v=np.array([[0,0,z0],[side,0,z0],[side,side,z0],[0,side,z0],
                    [0,0,z1],[side,0,z1],[side,side,z1],[0,side,z1]])
        faces=[[v[i] for i in f] for f in ([0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7])]
        ax.add_collection3d(Poly3DCollection(faces, facecolors=(*color,alpha), edgecolors='none'))

    def plane(zp, color=(0.2,0.2,0.2), alpha=0.25):
        v=np.array([[0,0,zp],[side,0,zp],[side,side,zp],[0,side,zp]])
        ax.add_collection3d(Poly3DCollection([v], facecolors=(*color,alpha), edgecolors='none'))

    # highlight zones: input=red, output=green
    slab(0.00,0.25,(1.0,0.0,0.0),alpha=0.03)   # input slab (red)
    slab(0.75,1.00,(0.0,0.8,0.0),alpha=0.03)   # output slab (green)
    plane(0.25, color=(1.0,0.0,0.0), alpha=0.05)   # divider between input & recurrent
    plane(0.75, color=(0.0,0.8,0.0), alpha=0.05)   # divider between recurrent & output

    scatter = ax.scatter(x,y,z, c=cols, s=point_size, edgecolor='k', linewidths=0.25)
    for a,b in zip(ii,jj):
        ax.plot([x[a],x[b]],[y[a],y[b]],[z[a],z[b]], lw=0.5, alpha=line_alpha, color='k')

    def update(f):
        ax.view_init(elev=15+0.1*f, azim=f); return scatter,
    ani = FuncAnimation(fig, update, frames=np.arange(0,720,2), interval=20, blit=False)
    return fig, ax, ani


class SpatialConnectivityManager:
    def __init__(self, N, num_inputs, num_outputs, device, dtype, generator,
                 neuronal_density=100.0, cdeg=12.0, spectral_rad=1.2, exc2inh=4.0,
                 i_zone=0.25, o_zone=0.75):
        self.N = N
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.nd = neuronal_density
        self.tdeg = cdeg * math.log(self.N + 1.0) / math.log(100 + 1.0)
        self.exc2inh = exc2inh
        self.device = device
        self.dtype = dtype
        self.generator = generator
        self.spectral_radius = spectral_rad
        self.i_zone = i_zone
        self.o_zone = o_zone

        Nf = torch.tensor(float(self.N), device=self.device, dtype=self.dtype)
        self.mu_E = 1.0 / torch.sqrt(Nf)
        self.mu_I = float(self.exc2inh) / torch.sqrt(Nf)
        self.std = 1.0 / torch.sqrt(Nf)

        self.positions, self.side_len = self.set_positions()
        self.in_input, self.in_output = self.set_zones()
        self.exc_bool, self.dale_mask = self.assign_dale()
        self.rec_mask_fn = self.set_mask_fn()

    def set_positions(self):
        Nf = torch.tensor(float(self.N), device=self.device, dtype=self.dtype)
        side_len = torch.sqrt(Nf / float(self.nd))
        positions = torch.rand(self.N, 3, generator=self.generator, device=self.device, dtype=self.dtype)
        positions[:, 1:] *= side_len
        return positions, side_len

    def set_zones(self):
        z = self.positions[:, 0]
        return (z < self.i_zone), (z >= self.o_zone)

    def set_mask_fn(self):
        x = self.positions[:, 1].unsqueeze(1)
        y = self.positions[:, 2].unsqueeze(1)
        z = self.positions[:, 0].unsqueeze(1)
        dist2 = (x - x.T)**2 + (y - y.T)**2 + (z - z.T)**2
        def make_mask(radius):
            M = dist2 <= (radius * radius)
            M.fill_diagonal_(False)
            return M
        return make_mask

    def get_recurrent_mask(self):
        r0 = ((3.0 * max(self.tdeg, 1e-6)) / (4.0 * math.pi * float(self.nd))) ** (1.0 / 3.0)
        r = self._calibrate_radius(r_init=r0)
        return self.rec_mask_fn(r)  # bool mask

    def get_mean_deg(self, r):
        return self.rec_mask_fn(r).sum(1).float().mean().item()

    def _calibrate_radius(self, r_init=0.05, it_bracket=16, it_bisect=16):
        r_lo = 0.0
        r_hi = r_init
        deg_hi = self.get_mean_deg(r_hi)
        for _ in range(it_bracket):
            if deg_hi >= self.tdeg and deg_hi > 0.0:
                break
            r_hi = max(1e-12, r_hi * 2.0)
            deg_hi = self.get_mean_deg(r_hi)
        if deg_hi <= 0.0 or deg_hi < self.tdeg:
            return r_hi
        for _ in range(it_bisect):
            r_mid = 0.5 * (r_lo + r_hi) if r_lo > 0.0 else 0.5 * r_hi
            deg_mid = self.get_mean_deg(r_mid)
            if deg_mid < self.tdeg:
                r_lo = r_mid
            else:
                r_hi = r_mid
        return r_hi

    def assign_dale(self):
        p_exc = float(self.exc2inh) / (1.0 + float(self.exc2inh))
        target_exc = int(round(p_exc * self.N))
        exc_bool = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        exc_bool[self.in_output] = True
        need_more = max(0, target_exc - int(exc_bool.sum().item()))
        pool = (~self.in_output).nonzero(as_tuple=False).flatten()
        if need_more > 0 and pool.numel() > 0:
            sel = pool[torch.randperm(pool.numel(), generator=self.generator, device=self.device)[:need_more]]
            exc_bool[sel] = True
        dale_mask = (exc_bool.to(self.dtype) * 2.0 - 1.0)
        return exc_bool, dale_mask

    @staticmethod
    def spectral_normalize(W, target_rad, iters=20):
        v = torch.randn(W.size(1), device=W.device, dtype=W.dtype)
        v = v / (v.norm() + 1e-12)
        for _ in range(iters):
            v = W @ v
            n = v.norm()
            v = v / (n + 1e-12)
        s = (W @ v).norm()  # top singular value approx
        s = torch.clamp(s, min=1e-12)
        return (float(target_rad) / float(s)) * W

    def init_io_weights(self):
        W_inp = (self.mu_E + self.std * torch.randn(self.N, self.num_inputs, generator=self.generator, device=self.device, dtype=self.dtype)).abs()
        W_out = (self.mu_E + self.std * torch.randn(self.num_outputs, self.N, generator=self.generator, device=self.device, dtype=self.dtype)).abs()
        W_inp = W_inp * self.in_input.unsqueeze(1).expand(self.N, self.num_inputs).to(self.dtype)
        W_out = W_out * self.in_output.unsqueeze(0).expand(self.num_outputs, self.N).to(self.dtype)
        return W_inp, W_out

    def init_rec_weights(self):
        mag_E = (self.mu_E + self.std * torch.randn(self.N, self.N, generator=self.generator, device=self.device, dtype=self.dtype)).abs()
        mag_I = (self.mu_I + self.std * torch.randn(self.N, self.N, generator=self.generator, device=self.device, dtype=self.dtype)).abs()
        Ecol = self.exc_bool.unsqueeze(0).expand_as(mag_E)
        W_rec = torch.where(Ecol, mag_E, -mag_I)
        W_rec = (W_rec * self.recurrent_mask.to(self.dtype)).fill_diagonal_(0.0)
        return W_rec

    def get_connectivity(self):
        self.recurrent_mask = self.get_recurrent_mask()
        W_inp, W_out = self.init_io_weights()
        self.input_mask = (W_inp != 0)
        self.output_mask = (W_out != 0)
        W_rec = self.init_rec_weights()
        W_rec = self.spectral_normalize(W_rec, self.spectral_radius)
        return (W_rec, W_inp, W_out,
                self.recurrent_mask.to(self.dtype), self.output_mask.to(self.dtype),
                self.input_mask.to(self.dtype), self.dale_mask, self.positions)

class RNN_torch(torch.nn.Module):
    def __init__(self,
                 N,
                 activation_name='relu',
                 activation_slope=1.0,
                 dt=1.0,
                 tau=10.0,
                 exc_to_inh_ratio=4.0,
                 neuronal_density=100.0,
                 spectral_rad=1.2,
                 cdeg=12.0,
                 sigma_rec=0.05,
                 sigma_inp=0.05,
                 gamma=0.1,
                 d=0.0,
                 y_init=None,
                 seed=None,
                 input_size=6,
                 output_size=2):
        super().__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # self.device = torch.device('cpu')
        self.dtype = torch.get_default_dtype()
        self.N = int(N)
        self.dt, self.tau = float(dt), float(tau)
        self.alpha = torch.tensor(self.dt / self.tau, device=self.device, dtype=self.dtype)

        # scalars on-device
        self.sigma_rec = torch.tensor(float(sigma_rec), device=self.device, dtype=self.dtype)
        self.sigma_inp = torch.tensor(float(sigma_inp), device=self.device, dtype=self.dtype)
        self.gamma = torch.tensor(float(gamma), device=self.device, dtype=self.dtype)
        self.d = torch.tensor(float(d), device=self.device, dtype=self.dtype)

        # sizes / params
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.spectral_rad = float(spectral_rad)
        self.neuronal_density = float(neuronal_density)
        self.exc2inhR = float(exc_to_inh_ratio)
        self.cdeg = float(cdeg)

        # activation
        slope = float(activation_slope)
        act_map = {
            "relu":     (lambda x: torch.relu(slope * x)),
            "tanh":     (lambda x: torch.tanh(slope * x)),
            "sigmoid":  (lambda x: torch.sigmoid(slope * x)),
            "softplus": (lambda x: torch.nn.Softplus(beta=slope)(x)),
        }
        self.activation_name = activation_name
        self.activation_slope = slope
        self.activation = act_map[self.activation_name]

        # state init
        self.y_init = (y_init.to(self.device, self.dtype) if isinstance(y_init, torch.Tensor)
                       else torch.zeros(self.N, device=self.device, dtype=self.dtype))
        self.last_dropout_mask = torch.ones(self.N, device=self.device, dtype=self.dtype)

        # RNG
        self.random_generator = torch.Generator(device=self.device)
        self.random_generator.manual_seed(int(seed) if seed is not None else int(torch.randint(10**9, (1,), generator=self.random_generator).item()))

        # connectivity
        self.con_manager = SpatialConnectivityManager(
            N=self.N,
            num_inputs=self.input_size,
            num_outputs=self.output_size,
            device=self.device,
            dtype=self.dtype,
            generator=self.random_generator,
            neuronal_density=self.neuronal_density,
            cdeg=self.cdeg,
            spectral_rad=self.spectral_rad,
            exc2inh=self.exc2inhR
        )
        W_rec, W_inp, W_out, self.recurrent_mask, self.output_mask, self.input_mask, self.dale_mask, self.positions = \
            self.con_manager.get_connectivity()

        # parameters on device
        self.W_rec = torch.nn.Parameter(W_rec.to(self.device, self.dtype))
        self.W_inp = torch.nn.Parameter(W_inp.to(self.device, self.dtype))
        self.W_out = torch.nn.Parameter(W_out.to(self.device, self.dtype))

        # # optional visualization if available
        # if 'visualize_connectivity_3d' in globals():
        #     _ = visualize_connectivity_3d(self.W_rec, self.W_inp, self.W_out,
        #                                   self.recurrent_mask, self.dale_mask,
        #                                   self.output_mask, self.input_mask, self.positions,
        #                                   d=1.0, max_edges=4000, point_size=18, line_alpha=0.15)
        #     if 'plt' in globals(): plt.show()

    def rhs(self, s, I, i_noise, r_noise, dropout_mask=None):
        s_do = s if dropout_mask is None else s * dropout_mask  # (N,B) · (N,1)
        h = self.W_rec @ s_do + self.W_inp @ (I + i_noise)
        return - (s - self.d) + self.activation(h) + r_noise - self.gamma * s ** 3

    def adversarial_dropout_mask(self, participation, drop_rate, adv=5.0,
                                 mass_preserve=True, eps=1e-8, cap=1.25):
        N = participation.numel()
        K = int(round(drop_rate * N))
        K = max(0, min(N - 1, K))
        if K == 0:
            return torch.ones(N, 1, device=participation.device, dtype=participation.dtype)
        with torch.no_grad():
            p = (participation / participation.max().clamp_min(eps)).detach()
            s = torch.softmax(adv * p, dim=0)
            idx_drop = torch.multinomial(s, num_samples=K, replacement=False)
            keep = torch.ones(N, device=p.device, dtype=p.dtype)
            keep[idx_drop] = 0.0
            if mass_preserve:
                Pk = (p * keep).sum()
                Pd = (p * (1.0 - keep)).sum()
                c = 1.0 + Pd / (Pk + eps)
                m = (keep * c).unsqueeze(-1)
            else:
                m = keep.unsqueeze(-1)
            if cap is not None:
                m = m.clamp_max(cap)
        return m

    def forward(self, u, w_noise=True, dropout=False, drop_rate=0.05, participation=None):
        T_steps, batch_size = u.shape[1], u.shape[-1]
        train_do = dropout and (drop_rate > 0)
        do_mask = self.adversarial_dropout_mask(participation, drop_rate) if train_do else None  # (N,1)

        states = torch.zeros(self.N, 1, batch_size, device=self.device)
        states[:, 0, :] = self.y_init.reshape(-1, 1).repeat(1, batch_size)

        rec_noise = torch.zeros(self.N, T_steps, batch_size, device=self.device)
        inp_noise = torch.zeros(self.input_size, T_steps, batch_size, device=self.device)
        if w_noise:
            rec_noise = torch.sqrt((2 / self.alpha) * self.sigma_rec ** 2) * torch.randn(
                *rec_noise.shape, generator=self.random_generator, device=self.device
            )
            inp_noise = torch.sqrt((2 / self.alpha) * self.sigma_inp ** 2) * torch.randn(
                *inp_noise.shape, generator=self.random_generator, device=self.device
            )

        states_list = [states[:, 0, :]]
        for t in range(1, T_steps):
            rhs_val = self.rhs(
                s=states_list[-1],
                I=u[:, t - 1, :],
                i_noise=inp_noise[:, t - 1, :],
                r_noise=rec_noise[:, t - 1, :],
                dropout_mask=do_mask
            )
            states_list.append(states_list[-1] + self.alpha * rhs_val)

        states_new = torch.stack(states_list, dim=1)  # (N, T, B)
        with torch.no_grad():
            do_mask = torch.ones(self.N, 1, 1, device=self.device) if do_mask is None else do_mask
            unscaled_do_mask = do_mask.float().clamp(0, 1).reshape(self.N, 1, 1)
        outputs = torch.einsum("oj,jtk->otk", self.W_out, states_new * unscaled_do_mask)
        return states_new, outputs

    def get_params(self):
        '''
        Save crucial parameters of the RNN as numpy arrays
        :return: parameter dictionary containing connectivity parameters, initial conditions,
         number of nodes, dt and tau
        '''
        param_dict = {}
        W_out = deepcopy(self.W_out.data.cpu().detach().numpy())
        W_rec = deepcopy(self.W_rec.data.cpu().detach().numpy())
        W_inp = deepcopy(self.W_inp.data.cpu().detach().numpy())
        y_init = deepcopy(self.y_init.detach().cpu().numpy())
        param_dict["activation_name"] = self.activation_name
        param_dict["activation_slope"] = self.activation_slope
        param_dict["W_out"] = W_out
        param_dict["W_inp"] = W_inp
        param_dict["W_rec"] = W_rec
        param_dict["output_mask"] = self.output_mask.detach().cpu().numpy()
        param_dict["recurrent_mask"] = self.recurrent_mask.detach().cpu().numpy()
        param_dict["input_mask"] = self.input_mask.detach().cpu().numpy()
        param_dict["y_init"] = y_init
        param_dict["N"] = self.N
        param_dict["dt"] = self.dt
        param_dict["tau"] = self.tau
        param_dict["gamma"] = self.gamma
        param_dict["d"] = self.d
        param_dict["positions"] = self.positions.detach().cpu().numpy()
        param_dict["dale_mask"] = self.dale_mask.detach().cpu().numpy()
        return param_dict

    def set_params(self, params):
        def to_dev(x, dtype=None):
            if isinstance(x, torch.Tensor):
                t = x.detach().clone()
                return t.to(device=self.device, dtype=dtype if dtype is not None else t.dtype)
            return torch.as_tensor(x, device=self.device, dtype=dtype)

        self.N = params["W_rec"].shape[0]

        # matrices
        self.W_out.data = to_dev(params["W_out"], torch.float32)
        self.W_inp.data = to_dev(params["W_inp"], torch.float32)
        self.W_rec.data = to_dev(params["W_rec"], torch.float32)
        self.output_mask = to_dev(params["output_mask"], torch.float32)
        self.recurrent_mask = to_dev(params["recurrent_mask"], torch.float32)
        self.input_mask = to_dev(params["input_mask"], torch.float32)

        # scalars / vectors
        self.gamma = to_dev(params["gamma"])
        self.d = to_dev(params["d"])
        self.y_init = to_dev(params["y_init"], torch.float32)
        self.activation_slope = to_dev(params["activation_slope"])
        self.positions = to_dev(params["positions"], torch.float32)
        self.dale_mask = to_dev(params["dale_mask"], torch.float32)
        return None


if __name__ == '__main__':
    N = 300
    activation_name = 'tanh'
    rnn_torch = RNN_torch(N=N, activation_name=activation_name)
    param_dict = rnn_torch.get_params()
    # print(param_dict)
