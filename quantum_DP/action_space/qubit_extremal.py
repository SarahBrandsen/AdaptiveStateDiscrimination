import numpy as np
import torch


class Qubit_Extremal_POVM_ParamSpace:
    '''
    Action space for qubit extremal measurements with three matrices
        alpha1 |phi1><phi1| + alpha2 |phi2><phi2| + alpha3 |phi3><phi3| = I
    In the format
        {'1,2,3': {'+': alpha1 |phi1><phi1|, '-': alpha2 |phi2><phi2|, '*': alpha3 |phi3><phi3|}}
        where alpha1 >= alpha2 >= alpha3
    '''

    def __init__(self, Qalpha=100, Qphi=20, device='cpu'):
        self.device = device

        # Find all alpha tuples satisfying:
        #     alpha1 >= alpha2 >= alpha3
        #     alpha1 + alpha2 + alpha3 = 2
        a = np.linspace(0, 1, Qalpha + 1)
        I, J = np.where((a >= 1 / 2) & (a[:, None] >= np.maximum(a, 2 * (1 - a))))

        # For each alpha tuple, there are 2 * Qphi phi tuples satisfying:
        # I = alpha1 * outer(phi1) + alpha2 * outer(phi2) + alpha3 * outer(phi3)
        alpha1, alpha2, alpha3 = [a.repeat(2 * Qphi) for a in [a[I], a[J], 2 - a[I] - a[J]]]

        # For each alpha tuple, compute corresponding theta and beta
        theta = np.arcsin(np.sqrt((1 - alpha3) / (alpha1 * alpha2)))
        beta = np.arcsin((alpha2 * (alpha3 != 0)) / (alpha3 + (alpha3 == 0)) * np.sin(2 * theta))

        # For each alpha tuple and phi1, compute corresponding phi2 and phi3
        phi1 = np.tile(np.linspace(0, np.pi, Qphi + 1)[:-1], len(I)).repeat(2)
        phi2 = phi1 + theta * (-1)**np.arange(len(theta))
        phi3 = phi1 + np.pi / 2 + beta / 2 * (-1)**np.arange(len(beta))

        param_space = np.vstack([alpha1, phi1, alpha2, phi2, alpha3, phi3]).T
        self.param_space = torch.tensor(param_space).double().to(self.device)

        # Generate projective measurements
        self.outcomes = ['+', '-', '*']
        self.allocations = ['1,2,3']
        self.proj_meas = {'1,2,3': {
            '+': self.construct(alpha1, phi1),
            '-': self.construct(alpha2, phi2),
            '*': self.construct(alpha3, phi3)
        }}

        # Assert the constraint holds
        assert torch.allclose(torch.eye(2).double().to(self.device), sum(v for v in self.proj_meas['1,2,3'].values()))

    def construct(self, alpha, phi):
        '''
        Construct rank-1 matrix given alpha and phi
        Arguments:
            alpha {number} -- amplitude
            phi {number} -- angle
        Returns:
            torch.DoubleTensor -- corresponding matrix in the extremal measurement
        '''
        vec = np.vstack([np.cos(phi), np.sin(phi)]).T
        mats = np.einsum('r, ri, rj->rij', alpha, vec, vec)
        return torch.tensor(mats).double().to(self.device)

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
        alpha1, phi1, alpha2, phi2, alpha3, phi3 = A_param
        M = {
            '+': alpha1 * np.outer(*[[np.cos(phi1), np.sin(phi1)]] * 2),
            '-': alpha2 * np.outer(*[[np.cos(phi2), np.sin(phi2)]] * 2),
            '*': alpha3 * np.outer(*[[np.cos(phi3), np.sin(phi3)]] * 2)
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
