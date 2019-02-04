import numpy as np
import torch
from collections import deque


class DP_Node:
    ''' A node in the computation graph of quantum DP
    Indexed by the set of remaining qubit indices: S
    Stores
        - the optimal error probability function: R
        - optimal choice of next qubit: A_index
        - optimal action parameter for next measurement: A_param
        - optimal allocation to form matrices in measurement: A_allot
    '''

    def __init__(self, S, Qp, param_dim, only_best_order, interp_mode, device):
        '''
        Initialize a DP node
        Arguments:
            S {set} -- set of remaining qubit indices
            Qp {int} -- quantization resolution of p in [0, 1]
            param_dim {int} -- number of parameters to determine an action (before allocation)
            only_best_order {bool} -- whether only store the best ordering and ignore worst ordering
            interp_mode {str} -- interpolation mode, can be {zero, nearest, linear, cubic}
            device {str} -- name of working device
        '''
        # Initial the node to be not computed and children dictionary
        self.computed, self.children = False, {}
        # Set working device and interpolation mode
        self.device, self.interp_mode = device, interp_mode or 'linear'
        assert self.interp_mode in ['zero', 'nearest', 'linear', 'cubic']
        # Set DP node index and resolution
        self.S, self.Qp = np.array(S), Qp
        # Initialize storage variables
        if only_best_order:
            # Only keep best order
            self.R = torch.ones(1, Qp + 1).double().to(self.device)
            self.A_index = torch.zeros(1, Qp + 1).long().to(self.device)
            self.A_param = torch.zeros(1, Qp + 1, param_dim).double().to(self.device)
            self.A_allot = torch.zeros(1, Qp + 1).long().to(self.device)
        else:
            # Keep best order and worst order
            self.R = torch.stack([torch.ones(Qp + 1), torch.zeros(Qp + 1)]).double().to(self.device)
            self.A_index = torch.zeros(2, Qp + 1).long().to(self.device)
            self.A_param = torch.zeros(2, Qp + 1, param_dim).double().to(self.device)
            self.A_allot = torch.zeros(2, Qp + 1).long().to(self.device)

    def p_index(self, p):
        '''
        Given probability p, get index using nearest criterion
        Arguments:
            p {number} -- input probabilities
        Returns:
            torch.LongTensor -- corresponding indices of the input probabilities
        '''
        idx = torch.tensor(p * self.Qp + 0.5).long()
        return idx.to(self.device)

    def get_R(self, hidx, hdel):
        '''
        Given the interval index and fraction to left end, return interpolated R values
        Arguments:
            hidx {torch.LongTensor} -- interval index for each input probability
            hdel {torch.DoubleTensor} -- fraction to left end for each input probability
        Returns:
            torch.DoubleTensor -- interpolated optimal error probability
        '''
        if self.interp_mode == 'zero':
            # 0-th order interpolation
            return self.R[:, hidx]
        elif self.interp_mode == 'nearest':
            # nearest neighbor interpolation
            return self.R[:, hidx + (hdel >= 0.5).long()]
        elif self.interp_mode == 'linear':
            # linear interpolation
            first = self.R[:, hidx]
            second = self.R[:, hidx + (hidx < self.Qp).long()]
            return (1 - hdel) * first + hdel * second
        elif self.interp_mode == 'cubic':
            # cubic splint interpolation
            if not hasattr(self, 'a') or not hasattr(self, 'b'):
                # if is called for the first time, do the cubic spline pre-computation
                Qp, y = self.Qp, self.R.t()
                dy = torch.cat([y[1:], y[[-1]]]) - torch.cat([y[[0]], y[:-1]])
                M = (torch.diag(torch.ones(Qp), 1) + torch.diag(torch.ones(Qp), -1) + 2 * torch.diag(1 + (torch.arange(Qp + 1) % Qp != 0).float())).double().to(self.device)
                k = 3 * Qp * M.inverse() @ dy
                self.a = (k[:-1] / Qp - (y[1:] - y[:-1]))
                self.a = torch.cat([self.a, self.a[[-1]]]).t()
                self.b = (-k[1:] / Qp + (y[1:] - y[:-1]))
                self.b = torch.cat([self.b, self.b[[-1]]]).t()

            hh = (1 - hdel) * hdel
            first = self.R[:, hidx] + hh * self.a[:, hidx]
            second = self.R[:, hidx + (hidx < self.Qp).long()] + hh * self.b[:, hidx]
            return (1 - hdel) * first + hdel * second

    def prob_success(self, p):
        '''
        Given prior probability p, get the corresponding success probability
        Arguments:
            p {number} -- input prior probabilities
        Returns:
            np.array -- corresponding success probabilities
        '''
        p = torch.tensor(p).double().to(self.device)
        hidx = (p * self.Qp).long()
        hdel = (self.Qp * (p - hidx.double() / self.Qp)).clamp(1e-20, 1 - 1e-20)
        return (1 - self.get_R(hidx, hdel)).to('cpu', dtype=torch.float64).numpy()

    def best_next_action(self, p):
        '''
        Given prior probability p, get the corresponding best action
        Arguments:
            p {number} -- input prior probabilities
        Returns:
            (np.array, np.array, np.array) -- corresponding best next qubit index, action parameter, allocation
        '''
        idx = self.p_index(p)
        A_index = self.A_index[:, idx].to('cpu', dtype=torch.int64).numpy()
        A_param = self.A_param[:, idx].to('cpu', dtype=torch.float64).numpy()
        A_allot = self.A_allot[:, idx].to('cpu', dtype=torch.int64).numpy()
        return A_index, A_param, A_allot


class Quantum_DP:
    '''
    Quantum DP class handling all computations
    '''

    def __init__(self, N, rho_pos, rho_neg,
                 param_space, Qp=100, interp_mode='linear',
                 cache=False, device='cpu'):
        '''
        Initialize the Quantum DP object
        Arguments:
            N {int} -- number of qibuts
            rho_pos {np.array} -- array of rhohat_+ matrices
            rho_neg {np.array} -- array of rhohat_- matrices
            param_space -- action parameter space
        Keyword Arguments:
            Qp {number} -- quantization resolution of p in [0, 1] (default: {100})
            interp_mode {str} -- interpolation mode (default: {'linear'})
            cache {bool} -- whether to cache intermediate results (default: {False})
            device {str} -- name of working device (default: {'cpu'})
        '''
        self.N, self.cache, self.device = N, cache, device
        # Parse rho_pos and rho_neg, check if any of them are identical
        self.rho = self.parse_states(rho_pos, rho_neg)
        # Quantize [0, 1] to Qp + 1 points
        self.Qp, self.p_vals = Qp, torch.linspace(0, 1, Qp + 1).double().to(self.device)

        # Set action parameter space and initialize cache (if cache is True)
        self.Asp = param_space
        if self.cache:
            self.g, self.h, self.hidx, self.hdel = [{
                tau: {
                    d: [None] * N for d in self.Asp.outcomes
                } for tau in self.Asp.allocations
            } for _ in range(4)]

        # Build computation graph
        self.root = self.build_graph(interp_mode)
        # Run DP computation
        self.compute_DP(self.root)

    def parse_states(self, rho_pos, rho_neg):
        '''
        Parse qubit states, detect identical copies
        Arguments:
            rho_pos {np.array} -- array of rhohat_+ matrices
            rho_neg {np.array} -- array of rhohat_- matrices
        Returns:
            dict[str: torch.DoubleTensor] -- dictionary of rhohat matrices on working device
        '''
        # Mapping from rhohat_+ and rhohat_- to qubit index
        rho_to_index = {}
        # Mapping from qubit index to the first occurrence index of its identical copy
        self.qubit_mapping = {}

        # Parse qubit states
        for i, (rp, rn) in enumerate(zip(rho_pos, rho_neg)):
            key = (tuple(rp.reshape(-1)), tuple(rn.reshape(-1)))
            if key not in rho_to_index:
                rho_to_index[key] = i
            self.qubit_mapping[i] = rho_to_index[key]

        # Check if all the qubits are identical copies
        self.all_identical = len(set(self.qubit_mapping.values())) == 1

        def to_tensor(x):
            if not np.iscomplexobj(x):
                return torch.tensor(x).double().to(self.device)
            else:
                return {
                    'r': torch.tensor(x.real).double().to(self.device),
                    'i': torch.tensor(x.imag).double().to(self.device)
                }
        return {
            '+': [to_tensor(m) for m in rho_pos],
            '-': [to_tensor(m) for m in rho_neg]
        }

    def build_graph(self, interp_mode):
        '''
        Build the computation graph for quantum DP
        Arguments:
            interp_mode {str} -- interpolation mode for the DP nodes
        Returns:
            DP_node -- the root node in the computation graph
        '''
        def make_key(S, j=-1):
            '''
            Make key by removing the first j in S and map each remained index to its first occurrence index
            Arguments:
                S {list[number]} -- list of qubit indices
            Keyword Arguments:
                j {number} -- index of excluded qubit, if -1 then all qubits will be included (default: {-1})
            Returns:
                tuple[number] -- key of DP node
            '''
            SS = list(S)
            if j in SS:
                SS.remove(j)
            ans = [self.qubit_mapping[jj] for jj in SS]
            return tuple(sorted(ans))

        # Set keyword arguments for DP node
        kwargs = {
            'Qp': self.Qp,
            'param_dim': self.Asp.param_space.shape[-1],
            'interp_mode': interp_mode,
            'only_best_order': self.all_identical,
            'device': self.device
        }

        # BFS construction of computation graph
        root_key = make_key(range(self.N))
        root = DP_Node(root_key, **kwargs)
        queue, cache = deque([root_key]), {root_key: root}
        while queue:
            node = cache.pop(queue.popleft())
            for j, key in [(self.qubit_mapping[j], make_key(node.S, j)) for j in node.S]:
                if key not in cache:
                    cache[key] = DP_Node(key, **kwargs)
                    queue.append(key)
                node.children[j] = cache[key]
        return root

    def compute_subcase(self, tau, d, j):
        '''
        Do sub-case computation for qubit j, allocation tau, and outcome d
        Arguments:
            tau {str} -- measurement allocation
            d {str} -- measurement outcome]
            j {int} -- index of measured qubit
        Returns:
            tuple(torch.tensor) -- (g, h, hidx, hdel), each of shape (Asp.param_space.shape[0], Qp + 1)
        '''
        if self.cache and self.g[tau][d][j] is not None:
            # If cached, directly grab the stored values
            g = self.g[tau][d][j]
            h = self.h[tau][d][j]
            hidx = self.hidx[tau][d][j]
            hdel = self.hdel[tau][d][j]
        else:
            t1 = self.Asp.meas_prob(tau, d, self.rho['+'][j], j) * self.p_vals.reshape(1, -1) + 1e-20
            t2 = self.Asp.meas_prob(tau, d, self.rho['-'][j], j) * (1 - self.p_vals).reshape(1, -1) + 1e-20

            # The action is determined by M_\tau
            # g, h, hidx, hdel are of shape (Asp.param_space.shape[0], Qp + 1)
            g = t1 + t2     # Likelihood, g_j(M_\tau, d, p)
            h = t1 / g      # Posterior, h_j(M_\tau, d, p)
            hidx = torch.tensor(h * self.Qp).long().to(self.device)
            hdel = (self.Qp * (h - self.p_vals[hidx])).clamp(1e-20, 1 - 1e-20)

            # Assert all values are valid
            assert not torch.isnan(g).any() and (g >= 0).all()
            assert (h >= 0).all() and (h <= 1).all()
            assert (hidx >= 0).all() and (hidx <= self.Qp).all()
            assert (hdel >= 0).all() and (hdel <= 1).all()

            # Cache the computed values if cache = True
            if self.cache:
                self.g[tau][d][j] = g
                self.h[tau][d][j] = h
                self.hidx[tau][d][j] = hidx
                self.hdel[tau][d][j] = hdel

        return g, h, hidx, hdel

    def compute_DP(self, node):
        '''
        DP computation for a DP node
        Arguments:
            node {DP_node} -- DP node to be computed
        '''
        if node.computed:
            # If the node has been computed, do nothing
            return
        if len(node.S) == 0:
            # Base case
            node.R = torch.min(self.p_vals, 1 - self.p_vals)
            node.R = node.R.repeat(2 - self.all_identical, 1)
        else:
            # Compute for each possible next index
            for j, ch in node.children.items():
                self.compute_DP(ch)
                # For each next index, compute the best error probability
                R_j = torch.ones_like(node.R)
                A_param_j = torch.zeros_like(node.A_param)
                A_allot_j = torch.zeros_like(node.A_allot)
                for t, tau in enumerate(self.Asp.allocations):
                    # compute \sum_d g_j(d, M_\tau, p) R_{S\j}( h_j(d, M_\tau, p) )
                    gen = (self.compute_subcase(tau, d, j) for d in self.Asp.outcomes)
                    temp = sum(g * ch.get_R(hidx, hdel) for g, h, hidx, hdel in gen)
                    R_temp, idx_temp = temp.min(dim=1)
                    update = R_temp < R_j
                    if update.any():
                        R_j[update] = R_temp[update]
                        A_param_j[update] = self.Asp.param_space[idx_temp[update]]
                        A_allot_j[update] = t
                # Best order: minimize over different next indices
                update = R_j[0] < node.R[0]
                if update.any():
                    node.R[0, update] = R_j[0, update]
                    node.A_index[0, update] = int(j)
                    node.A_param[0, update] = A_param_j[0, update]
                    node.A_allot[0, update] = A_allot_j[0, update]
                # Worst order: maximize over different next indices
                if self.all_identical:
                    continue
                update = R_j[1] > node.R[1]
                if update.any():
                    node.R[1, update] = R_j[1, update]
                    node.A_index[1, update] = int(j)
                    node.A_param[1, update] = A_param_j[1, update]
                    node.A_allot[1, update] = A_allot_j[1, update]
        # Mark current DP node as computed
        node.computed = True

    def get_node(self, S):
        '''
        Given set of index of remaining qubits, return the corresponding DP node
        Arguments:
            S {list[int]} -- set of index of remaining qubits
        Returns:
            DP_node -- corresponding DP node
        '''
        node, path = self.root, [i for i in range(self.N) if i not in S]
        for j in path:
            node = node.children[self.qubit_mapping[j]]
        return node
