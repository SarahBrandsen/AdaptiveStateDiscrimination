import numpy as np
from quantum_DP.mytqdm import tqdm_notebook_EX as tqdm
import os
import pickle
from itertools import product
from IPython.display import display
from ipywidgets import VBox
import xarray as xr


class ExperimentManager:
    def __init__(self, pickle_path, param_grid, num_trials):
        self.pickle_path = pickle_path
        self.param_grid = param_grid
        self.num_trials = num_trials
        self.load()

    def load(self):
        '''
        Load pickle file, if it does not exist, create one
        '''
        if not os.path.exists(self.pickle_path):
            with open(self.pickle_path, 'wb') as f:
                pickle.dump({}, f)
        with open(self.pickle_path, 'rb') as f:
            self.result = pickle.load(f)

    def save(self):
        '''
        Save current results pickle file
        '''
        with open(self.pickle_path, 'wb') as f:
            pickle.dump(self.result, f)

    def export_xarray(self, sel_func):
        '''
        Export the experiment result as an ND DataArray using xarray package
        Arguments:
            sel_func {function} -- scalar-valued function
        Returns:
            xr.DataArray -- ND DataArray
        '''
        names = [v[0] for v in self.param_grid]
        values = [v[1] for v in self.param_grid]
        shape = tuple(len(vals) for vals in values) + (self.num_trials, )
        data = np.empty(shape)
        for index in product(*[range(len(x)) for x in values]):
            key = tuple((name, vals[k]) for name, vals, k in zip(names, values, index))
            for t, res in enumerate(self.result[key]):
                data[index + (t,)] = sel_func(res)
        return xr.DataArray(data, coords=values + [list(range(self.num_trials))], dims=names + ['trial'])

    def run(self, simul_func, callback=lambda pbar, res: None):
        '''
        Run simul_func for given parameter grid and number of trials
        Arguments:
            simul_func {function} -- function for a single simulation
        Keyword Arguments:
            callback {function} -- callback to update the progress bar (default: {None})
        '''
        vbox = VBox()
        display(vbox)
        param_list = list(product(*[[(name, v) for v in vals] for name, vals in self.param_grid]))
        with tqdm(total=len(param_list), leave=False, vbox=vbox) as param_pbar:
            for params in param_list:
                param_pbar.set_postfix(**dict(params))
                self.result.setdefault(params, [])
                with tqdm(total=self.num_trials, leave=False, vbox=vbox) as trial_pbar:
                    trial_pbar.update(len(self.result[params]))
                    for t in range(len(self.result[params]), self.num_trials):
                        res = simul_func(**dict(params))
                        self.result[params].append(res)
                        callback(trial_pbar, self.result[params])
                        self.save()
                        trial_pbar.update(1)
                param_pbar.update(1)
