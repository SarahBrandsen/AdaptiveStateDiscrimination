# Adaptive State Discrimination
We develop adaptive techniques to approach quantum state discrimination over tensor product states. In particular, we consider a locally greedy adaptive scheme consisting of Bayesian updating of prior after each measurement and a dynamic-programming scheme which recursively minimizes expected future error.

#### Features

* GPU supported computation based on PyTorch.

* Smart construction of computation graph of quantum DP, avoid redundant computation over identical copies.

* Flexible design, users can implement custom `param_space` and feed in `Quantum_DP` instances to develop new Quantum DP algorithm.   

  Current builtin action spaces:

  * Locally greedy Helstrom action space
  * Qubit projective measurement action space
  * Qubit extremal POVM (with 3 outcomes) action space
  * Qutrit projective measurement action space

* Handy `ExperimentManager` class, a convenient tool to run experment over given parameter grid, save automatically to avoid losing result, and export result into `xarray` ND-Arrays.

#### Dependencies

* `torch` for GPU supported computation.
* `xarray` for multi-dimensional storage over a parameter grid.
* `ipywidgets`, `tqdm` for pretty progress bars in Jupyter notebook

#### How to Use

```python
from quantum_DP import *

# number of qubits
N = 7
# whether all qubits are identical copies
identical = False
# depolarizing coefficient
g = 0.3
# initial prior that state is '+'
q = 1/2
# resolution to quantize interval [0, 1] for Quantum DP
Qp = 100

# generate subsystems
rho_pos, rho_neg = generate_rhos(N, identical, g, dim=2)

# compute success probability of
#   1. global Helstrom measurement
#   2. local Helstrom measurement with majority vote
prob_succ_H = helstrom(q, rho_pos, rho_neg)

# create the action space
Asp = Locally_Greedy_ParamSpace(N, rho_pos, rho_neg, Qp, device)
# create Quantum DP instance
LG_QDP = Quantum_DP(N, rho_pos, rho_neg, Asp, Qp)
# get the expected success probability over optimial strategy
prob_succ_LG = LG_QDP.root.prob_success(q)

# print the result
if not identical:
    print('System with {0} different qubits'.format(N))
    print('Depolarizing parameter g = {0}'.format(g))
    print('Helstrom:\n    Global {0:.6f}\n    Local {1:.6f}'.format(*prob_succ_H))
    print('Locally greedy:\n    Best order  {0:.6f}\n    Worst order {1:.6f}'.format(
        *prob_succ_LG))
else:
    print('System with {0} identical qubits'.format(N))
    print('Depolarizing parameter g = {0}'.format(g))
    print('Helstrom:\n    Global {0:.6f}\n    Local {1:.6f}'.format(*prob_succ_H))
    print('Locally greedy   {0:.6f}'.format(*prob_succ_LG))
    
# simulate the optimal strategy for 1000 trials
simulate(LG_QDP, 1000, q);
```





