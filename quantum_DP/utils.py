import numpy as np
from functools import reduce
from quantum_DP.mytqdm import tqdm_notebook_EX as tqdm


def cum_kron(rhos):
    '''
    Cumulate Kronecker product
    Arguments:
        rhos {list[np.array]} -- list of matrices [A_1, A_2, ..., A_n]
    Returns:
        np.array -- A_1 otimes A_2 otimes ... otimes A_n
    '''
    return reduce(lambda cum, val: np.kron(cum, val), rhos, 1)


def helstrom(rho_pos, rho_neg):
    '''
    Success probability using Helstrom
    Arguments:
        rho_pos {list[np.array]} -- list of rhohat_+ matrices
        rho_neg {list[np.array]} -- list of rhohat_- matrices
    Returns:
        number -- success probability using Helstrom
    '''
    X = cum_kron(rho_pos) - cum_kron(rho_neg)
    return 1 / 2 + 1 / 4 * abs(np.linalg.eigvalsh(X)).sum()


def random_density_matrix(g, dim, is_complex=False):
    '''
    generate density operator of a uniform random pure state after depolarization
    Arguments:
        g {number} -- depolarizing coefficient
        dim {int} -- dimension of matrix
    Keyword Arguments:
        is_complex {bool} -- whether to generate complex matrix
    Returns:
        np.array -- density operator of a uniform random pure state after depolarization
    '''
    v = np.random.randn(dim)
    if is_complex:
        v = v + 1j * np.random.randn(dim)
    v = v / np.linalg.norm(v)
    return (1 - g) * np.outer(v, v.conj()) + g / dim * np.eye(dim)


def generate_rhos(N, identical, g, dim, is_complex=False):
    '''
    generate rhos
    Arguments:
        N {int} -- number of qubits
        identical {bool} -- whether the qubits are identical copies
        g {number} -- depolarizing coefficient
        dim {int} -- dimension of matrix
    Keyword Arguments:
        is_complex {bool} -- whether to generate complex matrix
    Returns:
        (np.array, np.array) -- rho_pos and rho_neg, with shape (N, dim, dim)
    '''
    if identical:
        return [
            np.tile(random_density_matrix(g, dim, is_complex), [N, 1, 1]),
            np.tile(random_density_matrix(g, dim, is_complex), [N, 1, 1])
        ]
    else:
        return [
            np.stack([random_density_matrix(g, dim, is_complex) for _ in range(N)]),
            np.stack([random_density_matrix(g, dim, is_complex) for _ in range(N)])
        ]


def simulate(qdp, trial, q=0.5):
    '''
    Simulate a quantum DP case for trial times
    Arguments:
        qdp {Quantum_DP} -- quantum DP object
        trial {int} -- number of trials
    Keyword Arguments:
        q {number} -- initial prior (default: {0.5})
    Returns:
        np.array -- measurement outcomes for each trial
    '''
    def to_numpy(x):
        return x.cpu().numpy() if not isinstance(x, dict) else x['r'].cpu().numpy() + 1j * x['i'].cpu().numpy()

    err = 0
    rhos = {'+': [to_numpy(x) for x in qdp.rho['+']], '-': [to_numpy(x) for x in qdp.rho['-']]}
    truth, outcomes, priors = np.empty(trial, dtype=np.str), np.empty((trial, qdp.N), dtype=np.str), np.empty((trial, qdp.N))
    for t in tqdm(range(trial), total=trial, leave=False):
        truth[t] = np.random.choice(['+', '-'], p=[q, 1 - q])
        rho_true = rhos[truth[t]]
        node, prior = qdp.root, q
        for n in range(qdp.N):
            j, A_param, A_allot = node.best_next_action(prior)
            j, A_param, A_allot = j[0], A_param[0], A_allot[0]
            rho, rho_pos, rho_neg = rho_true[j], rhos['+'][j], rhos['-'][j]
            d, prior = qdp.Asp.take_action(rho, rho_pos, rho_neg, prior, A_param, A_allot)
            node = node.children[j]
            outcomes[t, n], priors[t, n] = d, node.prob_success(prior)[0]
        err += (truth[t] == '+' and prior < 0.5) or (truth[t] == '-' and prior > 0.5)
    print('probability of success is', 1 - err / trial)
    return truth, outcomes, priors
