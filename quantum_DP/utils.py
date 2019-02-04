import numpy as np
from functools import reduce
from itertools import product
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


def helstrom(q, rho_pos, rho_neg):
    '''
    Success probability using Helstrom
    Arguments:
        q {number} -- prior probability that rho = rhohat_+
        rho_pos {list[np.array]} -- list of rhohat_+ matrices
        rho_neg {list[np.array]} -- list of rhohat_- matrices
    Returns:
        prob_global {number} -- success probability using Helstrom globally
        prob_local {number} -- success probability using Helstrom locally with majority vote
    '''
    # locally helstrom
    dist, N = {'+': [], '-': []}, len(rho_pos)
    for rp, rn in zip(rho_pos, rho_neg):
        lam, v = np.linalg.eigh((1 - q) * rn - q * rp)
        V = (lam >= 0) * v
        Pi_pos, Pi_neg = np.eye(lam.size) - V @ V.conj().T, V @ V.conj().T
        dist['+'].append([
            ('+', np.trace(Pi_pos @ rp)),
            ('-', np.trace(Pi_neg @ rp))
        ])
        dist['-'].append([
            ('+', np.trace(Pi_pos @ rn)),
            ('-', np.trace(Pi_neg @ rn))
        ])
    prob_local = q * sum(
        np.prod([v[1] for v in tup]) for tup in product(*dist['+'])
        if sum(t[0] == '+' for t in tup) > N / 2 - (q >= 1 / 2 and N % 2 == 0)
    ) + (1 - q) * sum(
        np.prod([v[1] for v in tup]) for tup in product(*dist['-'])
        if sum(t[0] == '-' for t in tup) > N / 2 - (q < 1 / 2 and N % 2 == 0)
    )

    # globally helstrom
    rho_pos, rho_neg = cum_kron(rho_pos), cum_kron(rho_neg)
    lam, v = np.linalg.eigh((1 - q) * rho_neg - q * rho_pos)
    V = (lam >= 0) * v
    Pi_neg, Pi_pos = V @ V.conj().T, np.eye(lam.size) - V @ V.conj().T
    prob_global = q * np.trace(Pi_pos @ rho_pos) + (1 - q) * np.trace(Pi_neg @ rho_neg)
    return prob_global.real, prob_local.real


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
