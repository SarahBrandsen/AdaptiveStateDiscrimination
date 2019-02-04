import numpy as np
import torch


class Locally_Greedy_ParamSpace:
    '''
    Action space for locally greedy
    '''

    def __init__(self, N, rho_pos, rho_neg, Qp, device='cpu'):
        '''
        Initialize the action space
        Arguments:
            N {int} -- number of qibuts
            rho_pos {np.array} -- array of rhohat_+ matrices
            rho_neg {np.array} -- array of rhohat_- matrices
            Qp {number} -- quantization resolution of p in [0, 1]
        Keyword Arguments:
            device {str} -- working device (default: {'cpu'})
        '''
        self.Qp, self.device = Qp, device
        self.rho = {
            '+': [self.to_tensor(m) for m in rho_pos],
            '-': [self.to_tensor(m) for m in rho_neg]
        }
        # Trivial parameter space
        self.param_space = torch.zeros(1, 1).double().to(self.device)
        Pi = [self.compute_Pi_matrix(rp, rn) for rp, rn in zip(rho_pos, rho_neg)]
        # Generate projective measurements
        self.outcomes = ['+', '-']
        self.allocations = ['1,2']
        self.proj_meas = {'1,2': {
            '+': [self.to_tensor(tup[0]) for tup in Pi],
            '-': [self.to_tensor(tup[1]) for tup in Pi]}
        }

    def compute_Pi_matrix(self, rho_pos, rho_neg):
        '''
        Compute Pi(p, j) matrix for all p
            Pi(p, j) = \sum_{|v>} |v><v|, where |v> is nonnegative eigenvector of (1-p) * rho_-^(j) - p * rho_+^(j)
        Arguments:
            rho_pos {np.array} -- rho_+^(j) matrix
            rho_neg {np.array} -- rho_-^(j) matrix
        Returns:
            (torch.DoubleTensor, torch.DoubleTensor) -- Pi(p, j) and I - Pi(p, j) matrices for all p given fixed j
        '''
        p = np.linspace(0, 1, self.Qp + 1).reshape(-1, 1, 1)
        lam, v = np.linalg.eigh((1 - p) * rho_neg - p * rho_pos)
        V = (lam[:, None, :] >= 0) * v
        res = V @ V.swapaxes(-2, -1).conj()
        return res, np.eye(2) - res

    def to_tensor(self, x):
        '''
        Convert np.array into torch.tensor
        Arguments:
            x {np.array} -- input np.array to be converted
        Returns:
            torch.DoubleTensor if x is real-valued
            dict[str:torch.DoubleTensor] if x is complex-valued
        '''
        if not np.iscomplexobj(x):
            return torch.tensor(x).double().to(self.device)
        else:
            return {
                'r': torch.tensor(x.real).double().to(self.device),
                'i': torch.tensor(x.imag).double().to(self.device)
            }

    def meas_prob(self, tau, d, rho, j):
        '''
        Compute Pr( d | rho, \mathcal{M}_\tau)
        Arguments:
            tau {str} -- measurement allocation
            d {str} -- measurement outcome
            rho {torch.DoubleTensor} -- density operators before measurement
        Returns:
            torch.DoubleTensor -- (param_space.shape[0],)
        '''
        # Compute Pi @ rho
        Pi = self.proj_meas[tau][d][j]
        if isinstance(rho, dict):
            val = Pi['r'] @ rho['r'] - Pi['i'] @ rho['i']
        else:
            val = Pi @ rho
        # Compute Tr(Pi @ rho)
        val = torch.einsum('imm->i', (val,)).clamp(0)
        # Round to 20 precision to avoid numerical issues
        val = ((val * 1e20).round() / 1e20).double()
        return val

    def take_action(self, rho, rho_pos, rho_neg, prior, A_param=None, A_allot=None):
        '''
        Given rhohat_+, rhohat_- and prior, rho is an unknown state can be
        either rhohat_+ or rhohat_-. Apply measurement parameterized by A_param
        and A_allot on rho, draw random outcome and the corresponding updated prior
        Arguments:
            rho {np.array} -- an unknown matrix that can be rhohat_+ or rhohat_-
            rho_pos {np.array} -- rhohat_+ state
            rho_neg {np.array} -- rhohat_- state
            prior {number} -- prior probability that rho is rhohat_+
        Keyword Arguments:
            A_param {None} -- unused for this case (default: {None})
            A_allot {None} -- unused for this case (default: {None})
        Returns:
            d {str} -- measurement outcome
            new_prior {number} -- the updated prior after the measurement
        '''
        # Construct measurement
        lam, v = np.linalg.eigh((1 - prior) * rho_neg - prior * rho_pos)
        V = (lam >= 0) * v
        Pi_pos, Pi_neg = V @ V.T, np.eye(2) - V @ V.T
        # Generate random outcome
        pmf = np.clip([np.trace(Pi_pos @ rho), np.trace(Pi_neg @ rho)], 0, 1)
        d = np.random.choice(['+', '-'], p=pmf)
        # Compute updated prior
        Pi = Pi_pos if d == '+' else Pi_neg
        t1 = np.trace(Pi @ rho_pos) * prior + 1e-20
        t2 = np.trace(Pi @ rho_neg) * (1 - prior) + 1e-20
        new_prior = t1 / (t1 + t2)
        return d, new_prior
