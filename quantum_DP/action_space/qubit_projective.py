import numpy as np
import torch


class Qubit_Proj_ParamSpace:
    '''
    Action space for qubit projective measurements
    In the format
        {'1,2': {'+': |phi><phi|, '-': |>psi<psi|}}
    where phi and psi are orthogonal.
    '''

    def __init__(self, Qphi=512, device='cpu'):
        '''
        Initialize the action space
        Keyword Arguments:
            Qphi {number} -- quantization resolution on phi (default: {512})
            device {str} -- working device (default: {'cpu'})
        '''
        self.device = device

        # The action space parameter is phi
        phi = np.linspace(0, np.pi / 2, Qphi + 1)[:-1].reshape(-1, 1)
        self.param_space = torch.tensor(phi).double().to(self.device)
        # For each phi compute an orthonormal basis for it
        self.ortho_basis, outers = self.orthonormal_basis(phi), {}
        for i in range(2):
            us = torch.tensor(self.ortho_basis[..., [i]]).double().to(self.device)
            outers[i + 1] = us @ us.transpose(-2, -1)
        # Generate projective measurements
        self.outcomes = ['+', '-']
        self.allocations = ['1,2']
        self.proj_meas = {'1,2': {'+': outers[1], '-': outers[2]}}

    def orthonormal_basis(self, phi):
        '''
        Given angle phi, create corresponding 2x2 rotation matrix
        Arguments:
            phi {number} -- given angle
        Returns:
            np.array -- rotation matrices
        '''
        c, s = np.cos(phi), np.sin(phi)
        u1, u2 = np.hstack([c, s]), np.hstack([-s, c])
        return np.stack([u1, u2], axis=2)

    def meas_prob(self, tau, d, rho, j):
        '''
        Compute Pr( d | rho, \mathcal{M}_\tau)
        Arguments:
            tau {str} -- measurement allocation
            d {str} -- measurement outcome
            rho {torch.DoubleTensor} -- density operators before measurement
            j [int] -- index of measured qubit
        Returns:
            torch.DoubleTensor -- (Asp.param_space.shape[0], 1)
        '''
        if isinstance(rho, dict):
            rho = rho['r']
        # Compute Pi @ rho
        val = self.proj_meas[tau][d] @ rho
        # Compute Tr(Pi @ rho)
        val = torch.einsum('imm->i', (val,)).clamp(0)
        # Round to 20 precision to avoid numerical issues
        val = ((val * 1e20).round() / 1e20).double()
        return val.reshape(-1, 1)

    def take_action(self, rho, rho_pos, rho_neg, prior, A_param, A_allot=None):
        '''
        Given rhohat_+, rhohat_- and prior, rho is an unknown state can be
        either rhohat_+ or rhohat_-. Apply measurement parameterized by A_param
        and A_allot on rho, draw random outcome and the corresponding updated prior
        Arguments:
            rho {np.array} -- an unknown matrix that can be rhohat_+ or rhohat_-
            rho_pos {np.array} -- rhohat_+ state
            rho_neg {np.array} -- rhohat_- state
            prior {number} -- prior probability that rho is rhohat_+
            A_param {np.array} -- action parameter
        Keyword Arguments:
            A_allot {None} -- unused for this case (default: {None})
        Returns:
            d {str} -- measurement outcome
            new_prior {number} -- the updated prior after the measurement
        '''
        # Construct measurement
        phi = A_param
        c, s = np.cos(phi), np.sin(phi)
        M = {
            '+': np.outer([c, s], [c, s]),
            '-': np.outer([-s, c], [-s, c])
        }
        # Generate random outcome
        pmf = np.clip([np.trace(M[d] @ rho).real for d in self.outcomes], 0, 1)
        d = np.random.choice(self.outcomes, p=pmf)
        # Compute updated prior
        Pi = M[d]
        t1 = np.trace(Pi @ rho_pos).real * prior + 1e-20
        t2 = np.trace(Pi @ rho_neg).real * (1 - prior) + 1e-20
        new_prior = t1 / (t1 + t2)
        return d, new_prior
