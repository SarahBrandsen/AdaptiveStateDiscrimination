import numpy as np
import torch


class Qutrit_Proj_ParamSpace:
    '''
    Action space for qutrit projective measurements
    First we generate an orthonormal basis {v1, v2, v3}
        - Format for ternary mode:
            {'1,2,3': {'+': |v1><v1|, '-': |v2><v2|, '*': |v3><v3| }}
        - Format for binary mode:
            {
                '12,3': {'+': |v1><v1| + |v2><v2|, '-': |v3><v3|},
                '13,2': {'+': |v1><v1| + |v3><v3|, '-': |v2><v2|},
                '23,1': {'+': |v2><v2| + |v3><v3|, '-': |v1><v1|}
            }
    '''

    def __init__(self, d=[2, 2, 2], Q=20, mode='ternary', device='cpu'):
        '''
        Initialize the action space
        Keyword Arguments:
            d {list} -- subdivisions of icosahedron (default: {[2, 2, 2]})
            Q {number} -- quantization resolution on equator (default: {20})
            mode {str} -- determines number of matrices in measurements (default: {'ternary'})
            device {str} -- working device (default: {'cpu'})
        '''
        assert mode in ['ternary', 'binary']
        self.mode, self.device, self.Q = mode, device, Q
        # Compute points of subdivision
        vertices, _ = self.icosahedron_subdivision(d)
        # Convert 3d-points on sphere into (phi, theta)
        phi, theta = self.euclidean_to_polar(vertices)
        # The action space parameter is (phi, theta, q)
        self.param_space = torch.tensor(
            np.vstack([
                phi.repeat(Q),
                theta.repeat(Q),
                np.tile(np.linspace(0, np.pi, Q + 1)[:-1], theta.size)
            ]).T
        ).double().to(self.device)
        # For each (phi, theta), compute Q different orthonormal basis for it
        self.ortho_basis, outers = self.orthonormal_basis(phi, theta, Q), {}
        for i in range(3):
            us = torch.tensor(self.ortho_basis[..., [i]]).to(self.device)
            outers[i + 1] = us @ us.transpose(-2, -1)
        # Generate projective measurements for possible allocations
        if self.mode == 'ternary':
            self.outcomes = ['+', '-', '*']
            self.allocations = ['1,2,3']
            self.proj_meas = {
                '1,2,3': {'+': outers[1], '-': outers[2], '*': outers[3]}
            }
        elif self.mode == 'binary':
            self.outcomes = ['+', '-']
            self.allocations = ['12,3', '13,2', '23,1']
            self.proj_meas = {
                '12,3': {'+': outers[1] + outers[2], '-': outers[3]},
                '13,2': {'+': outers[1] + outers[3], '-': outers[2]},
                '23,1': {'+': outers[2] + outers[3], '-': outers[1]}
            }

    def icosahedron_subdivision(self, subdiv):
        '''
        Create a subdivided icosahedron according to subdivision given by subdiv
        Arguments:
            subdiv {list[int]} -- subdivision number for each step, [d1, d2, ..., dk]
        Returns:
            vertices {np.array} -- vertex coordinates of the subdivided icosahedron
            faces {np.array} -- faces of subdivided icosahedron, each is a triple of vertex indices
        '''
        def mid_pt(i1, i2, m, n):
            '''
            Make a middle point p between vertex i1 and vertex i2,
            where the normalized dist(p, i1) / dist(i1, i2) = m / n
            '''
            key = (i1, i2, m, n) if i1 < i2 else (i2, i1, n - m, n)
            if key not in mid_pt_cache:
                p1, p2 = vertices[i1], vertices[i2]
                pt = (1 - m / n) * p1 + m / n * p2
                vertices.append(pt)
                mid_pt_cache[key] = len(vertices) - 1
            return mid_pt_cache[key]

        # Make the base icosahedron
        PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
        vertices = list(map(lambda pt: pt / np.linalg.norm(pt), [
            [-1, PHI, 0], [1, PHI, 0], [-1, -PHI, 0], [1, -PHI, 0],
            [0, -1, PHI], [0, 1, PHI], [0, -1, -PHI], [0, 1, -PHI],
            [PHI, 0, -1], [PHI, 0, 1], [-PHI, 0, -1], [-PHI, 0, 1],
        ]))
        faces = [
            # 5 faces around point 0
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            # Adjacent faces
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            # 5 faces around 3
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            # Adjacent faces
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ]

        # Subdivisions
        for n in subdiv:
            mid_pt_cache, faces_subdiv = {}, []
            for tri in faces:
                i1, i2, i3 = tri
                temp = [mid_pt(i1, i2, m, n) for m in range(1, n)]
                levels = [[i1] + temp + [i2]]
                for s in range(1, n):
                    j1 = mid_pt(i1, i3, s, n)
                    j2 = mid_pt(i2, i3, s, n)
                    temp = [mid_pt(j1, j2, m, n - s) for m in range(1, n - s)]
                    levels.append([j1] + temp + [j2])
                levels.append([i3])
                for l1, l2 in zip(levels, levels[1:]):
                    faces_subdiv += list(zip(l2, l1, l1[1:]))
                    faces_subdiv += list(zip(l2, l2[1:], l1[1:]))
            faces = faces_subdiv
            for key, i in mid_pt_cache.items():
                vertices[i] = vertices[i] / np.linalg.norm(vertices[i])
        return np.stack(vertices), np.stack(faces)

    def euclidean_to_polar(self, vertices):
        '''
        Given vertex coordinates on unit sphere, convert to polar angles
        Arguments:
            vertices {np.array} -- vertex coordinates on unit sphere
        Returns:
            (np.array, np.array) -- longitude and latitude angles
        '''
        phi = np.arctan2(vertices[:, 1], vertices[:, 0])
        theta = np.arccos(vertices[:, 2])
        return phi, theta

    def rotation_matrix(self, phi, theta):
        '''
        Given polar coordinates, compute the rotation matrix that maps north pole to the given point
        Arguments:
            phi {np.array} -- longitude
            theta {np.array} -- latitude
        Returns:
            np.array -- rotation matrix that maps north pole to the given point
        '''
        s1, c1 = np.sin(phi), np.cos(phi)
        s2, c2 = np.sin(theta), np.cos(theta)
        mats = np.array([
            [-s1, c1 * c2, c1 * s2],
            [c1, s1 * c2, s1 * s2],
            [np.zeros_like(theta), -s2, c2]
        ])
        if len(mats.shape) == 3:
            mats = mats.transpose([2, 0, 1])
        return mats

    def orthonormal_basis(self, phi, theta, Q):
        '''
        Given polar coordinate, return Q/2 orthonormal basis whose z axis aligns it
        Arguments:
            phi {np.array} -- longitude
            theta {np.array} -- latitude
            Q {int} -- quantization resolution on the equator
        Returns:
            np.array -- Q/2 orthonormal basis
        '''

        assert Q % 2 == 0
        M = self.rotation_matrix(phi, theta)
        omega = np.linspace(0, np.pi, Q + 1)[:-1]
        equator = np.vstack([np.cos(omega), np.sin(omega), np.zeros_like(omega)])
        ws = M @ equator

        u1 = ws.swapaxes(-1, -2)
        u2 = np.roll(ws, Q // 2, axis=-1).swapaxes(-1, -2)
        u3 = np.tile(M[..., [-1]], [1, Q]).swapaxes(-1, -2)
        # From (\phi_1, \theta_1, 0), ..., (\phi_1, \theta_1, Q-1), ...
        # to   (\phi_S, \theta_S, 0), ..., (\phi_S, \theta_S, Q-1)
        return np.vstack(np.stack([u1, u2, u3], axis=3))

    def meas_prob(self, tau, d, rho, j):
        '''
        Compute Pr( d | rho, \mathcal{M}_\tau)
        Arguments:
            tau {str} -- measurement allocation
            d {str} -- measurement outcome
            rho {torch.DoubleTensor} -- density operators before measurement
            j [int] -- index of measured qubit
        Returns:
            torch.DoubleTensor -- (param_space.shape[0], 1)
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

    def take_action(self, rho, rho_pos, rho_neg, prior, A_param, A_alloc):
        '''
        Given rhohat_+, rhohat_- and prior, rho is an unknown state can be
        either rhohat_+ or rhohat_-. Apply measurement parameterized by A_param
        and A_alloc on rho, draw random outcome and the corresponding updated prior
        Arguments:
            rho {np.array} -- an unknown matrix that can be rhohat_+ or rhohat_-
            rho_pos {np.array} -- rhohat_+ state
            rho_neg {np.array} -- rhohat_- state
            prior {number} -- prior probability that rho is rhohat_+
            A_param {np.array} -- action parameter
            A_alloc {int} -- allocation of orthonormal basis is self.allot[A_alloc]
        Returns:
            d {str} -- measurement outcome
            new_prior {number} -- the updated prior after the measurement
        '''
        # Construct measurement
        phi, theta, omega = A_param
        M = self.rotation_matrix(phi, theta)
        us = {
            1: M @ np.array([np.cos(omega), np.sin(omega), 0]),
            2: M @ np.array([-np.sin(omega), np.cos(omega), 0]),
            3: M @ np.array([0, 0, 1])
        }
        if self.mode == 'ternary':
            M = {
                '+': np.outer(us[1], us[1]),
                '-': np.outer(us[2], us[2]),
                '*': np.outer(us[3], us[3])
            }
        elif self.mode == 'binary':
            first, second = [[int(c) for c in s] for s in self.allocations[A_alloc].split(',')]
            M = {
                '+': sum(np.outer(us[i], us[i]) for i in first),
                '-': sum(np.outer(us[i], us[i]) for i in second)
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
