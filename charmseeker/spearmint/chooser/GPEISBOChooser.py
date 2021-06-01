import os
import gp
import util
import tempfile
import numpy as np
import numpy.random as npr
import scipy.linalg as spla
import scipy.stats as sps
import pickle

from helpers import *
from Locker import Locker

"""
Chooser module for Constrained Gaussian process expected improvement.
Candidates are sampled densely in the unit hypercube and then a subset
of the most promising points are optimized to maximize constrained EI
over hyper-parameter samples.  Slice sampling is used to sample
Gaussian process hyper-parameters for two GPs, one over the objective
function and the other a probit likelihood classification GP that estimates the
probability that a point is outside of the constraint space.
"""


def init_chooser(expt_dir, arg_string):
    args = util.unpack_args(arg_string)
    if 'budget' in args.keys():
        args['budget'] = float(args['budget'])
    return GPEISBOChooser(expt_dir, **args)


class GPEISBOChooser:

    def __init__(self, expt_dir, covar="Matern52", mcmc_iterations=20,
                 pending_samples=100, noiseless=False, burn_in=100, budget=1, cost_optimize=0):
        self.cov_func = getattr(gp, covar)
        self.locker = Locker()
        self.state_pkl = os.path.join(expt_dir, f'{self.__module__}.pkl')

        self.stats_file = os.path.join(expt_dir, f'{self.__module__}_hyper_parameters.txt')
        self.mcmc_iterations = mcmc_iterations
        self.burn_in = burn_in
        self.needs_burn_in = False
        self.pending_samples = pending_samples
        self.D = -1
        self.hyper_iterations = 1

        self.noiseless = bool(int(noiseless))
        self.constraint_noiseless = False
        self.hyper_samples = []
        self.constraint_hyper_samples = []

        self.noise_scale = 0.1  # horseshoe prior
        self.amp2_scale = 1    # zero-mean log normal prior
        self.max_ls = 2    # top-hat prior on length scales

        self.constraint_noise_scale = 0.1  # horseshoe prior
        self.constraint_amp2_scale = 1    # zero-mean log normal prior
        self.constraint_max_ls = 2   # top-hat prior on length scales

        self.budget = np.log(budget)  # convert to log scale
        self.cost_optimize = bool(cost_optimize)

    def get_budget(self):
        return self.budget

    def dump_hyper_params(self):
        self.locker.lock_wait(self.state_pkl)

        fh = tempfile.NamedTemporaryFile(mode='wb', delete=False)
        pickle.dump({'dims': self.D, 'ls': self.ls, 'amp2': self.amp2, 'noise': self.noise, 'mean': self.mean,
                     'constraint_ls': self.constraint_ls, 'constraint_amp2': self.constraint_amp2,
                     'constraint_noise': self.constraint_noise, 'constraint_mean': self.constraint_mean}, fh)
        fh.close()

        cmd = f'mv {fh.name} {self.state_pkl}'
        os.system(cmd)
        self.locker.unlock(self.state_pkl)

        fh = open(self.stats_file, 'w')
        fh.write('Mean Noise Amplitude <length scales>\n')
        fh.write('-----------ALL SAMPLES-------------\n')
        mean_hyper_params = 0 * np.hstack(self.hyper_samples[0])
        for i in self.hyper_samples:
            hyper_params = np.hstack(i)
            mean_hyper_params += (1/float(len(self.hyper_samples))) * hyper_params
            for j in hyper_params:
                fh.write(str(j) + ' ')
            fh.write('\n')

        fh.write('-----------MEAN OF SAMPLES-------------\n')
        for j in mean_hyper_params:
            fh.write(str(j) + ' ')
        fh.write('\n')
        fh.close()

    def _real_init(self, dims, values, costs):

        if not self.cost_optimize:
            # This is very important, to force calling _real_init when pass in a new budget
            cmd = f'rm {self.state_pkl}'
            os.system(cmd)

        self.locker.lock_wait(self.state_pkl)
        self.random_state = npr.get_state()

        if os.path.exists(self.state_pkl):
            fh = open(self.state_pkl, 'rb')
            state = pickle.load(fh)
            fh.close()

            self.D = state['dims']
            self.ls = state['ls']
            self.amp2 = state['amp2']
            self.noise = state['noise']
            self.mean = state['mean']
            self.constraint_ls = state['constraint_ls']
            self.constraint_amp2 = state['constraint_amp2']
            self.constraint_noise = state['constraint_noise']
            self.constraint_mean = state['constraint_mean']
        else:
            print("enter new _real_init")
            good_values = np.nonzero(np.logical_and(costs <= self.budget, np.isfinite(values)))[0]
            self.D = dims
            self.ls = np.ones(self.D)
            self.constraint_ls = np.ones(self.D)

            self.amp2 = np.std(values[good_values]) + 1e-4
            self.constraint_amp2 = np.std(costs) + 1e-4

            self.noise = 1e-3
            self.constraint_noise = 1e-3

            self.mean = np.mean(values[good_values])
            self.constraint_mean = np.mean(costs)

        self.needs_burn_in = False
        self.locker.unlock(self.state_pkl)

    def cov(self, amp2, ls, x1, x2=None):
        if x2 is None:
            return amp2 * (self.cov_func(ls, x1, None) + 1e-6 * np.eye(x1.shape[0]))
        else:
            return amp2 * self.cov_func(ls, x1, x2)

    def next(self, expt_grid, grid, values, costs, candidates, completes):
        # This function returns grid index to indicate the next configuration parameters
        # candidates is a numpy array with each indicate the index of a configuration in the grid
        # Cold start. There are less than 3 complete experiments
        if completes.shape[0] < 3:
            return int(candidates[0])

        complete_configs = grid[completes, :]
        candidate_configs = grid[candidates, :]
        complete_values = values[completes]
        complete_costs = costs[completes]

        # Convert from log values to normal scale
        print(f"current costs: {np.exp(complete_costs)}")
        print(f"self.budget: {np.exp(self.budget):.5f}")

        # Get feasible configuration indices
        idx = np.logical_and(complete_costs <= self.budget, np.isfinite(complete_values))
        good_values = np.nonzero(idx)[0]
        bad_values = np.nonzero(np.logical_not(idx))[0]
        print(f'Found {bad_values.shape[0]} constraint violating jobs')
        print(f'Received {good_values.shape[0]} valid results')

        # There are more than 3 completed experiments but the number of feasible
        # configurations is less than 2. This case is still identified as cold start.
        if good_values.shape[0] < 2:
            return int(candidates[0])

        if self.D == -1:
            # initialize gp parameters by evaluated samples
            self._real_init(grid.shape[1], complete_values, complete_costs)

        complete_list = expt_grid.get_complete_expt_params()

        if self.mcmc_iterations > 0:
            self.hyper_samples = []
            self.constraint_hyper_samples = []

            for _ in range(self.mcmc_iterations):
                self.sample_constraint_hyper_params(complete_configs, complete_costs)
                self.sample_hyper_params(complete_configs[good_values, :], complete_values[good_values])
            self.dump_hyper_params()

            overall_ei = self.ei_over_hyper_params(complete_configs, candidate_configs, complete_values, complete_costs)

            while True:
                best_candidate_index = np.argmax(np.mean(overall_ei, axis=1))
                grid_idx = int(candidates[best_candidate_index])
                params = []
                for param in expt_grid.get_params(grid_idx):
                    params.extend(param.int_val)
                params = list(map(int, params))
                print('params: ', params)
                if params not in complete_list:
                    return grid_idx
                overall_ei = np.delete(overall_ei, best_candidate_index, 0)
                candidates = np.delete(candidates, best_candidate_index, 0)
                print("delete!!!!!")
        else:
            print('This Chooser module permits only slice sampling with > 0 samples.')
            raise Exception('mcmc_iterations <= 0')

    def ei_over_hyper_params(self, complete_configs, candidate_configs, complete_values, complete_costs):
        overall_ei = np.zeros((candidate_configs.shape[0], self.mcmc_iterations))

        for mcmc_iter in range(self.mcmc_iterations):
            hyper = self.hyper_samples[mcmc_iter]
            constraint_hyper = self.constraint_hyper_samples[mcmc_iter]
            self.mean = hyper[0]
            self.noise = hyper[1]
            self.amp2 = hyper[2]
            self.ls = hyper[3]
            self.constraint_mean = constraint_hyper[0]
            self.constraint_noise = constraint_hyper[1]
            self.constraint_amp2 = constraint_hyper[2]
            self.constraint_ls = constraint_hyper[3]
            overall_ei[:, mcmc_iter] = self.compute_constrained_ei(complete_configs, candidate_configs,
                                                                   complete_values, complete_costs)
        return overall_ei

    def compute_constrained_ei(self, complete_configs, candidate_configs, complete_values, complete_costs):
        complete_full = complete_configs.copy()
        if not self.cost_optimize:
            complete_configs = complete_configs[complete_costs <= self.budget, :]
            complete_values = complete_values[complete_costs <= self.budget]
        return self.compute_constrained_ei_no_pending(complete_configs, complete_values, complete_full,
                                                      complete_costs, candidate_configs)

    def compute_constrained_ei_no_pending(self, complete_configs, complete_values, complete_full, complete_costs,
                                          candidate_configs):
        cons_comp_cov = self.cov(self.constraint_amp2, self.constraint_ls, complete_full)
        cons_cand_cross = self.cov(self.constraint_amp2, self.constraint_ls, complete_full, candidate_configs)

        cons_obsv_cov = cons_comp_cov + self.constraint_noise * np.eye(complete_full.shape[0])
        cons_obsv_chol = spla.cholesky(cons_obsv_cov, lower=True)

        t_alpha = spla.cho_solve((cons_obsv_chol, True), complete_costs - self.constraint_mean)
        t_beta = spla.solve_triangular(cons_obsv_chol, cons_cand_cross, lower=True)

        cons_func_m = np.dot(cons_cand_cross.T, t_alpha) + self.constraint_mean
        cons_func_v = self.constraint_amp2 * (1 + 1e-6) - np.sum(t_beta ** 2, axis=0)
        cons_func_s = np.sqrt(cons_func_v)

        if self.cost_optimize:
            best_cost = np.min(complete_costs)
            u = (best_cost - cons_func_m + 0.1) / cons_func_s
            n_cdf = sps.norm.cdf(u)
            n_pdf = sps.norm.pdf(u)
            return cons_func_s * (u * n_cdf + n_pdf)

        weight = sps.norm.cdf(self.budget, loc=cons_func_m, scale=cons_func_s)
        if len(complete_values) < 2:
            return weight

        best = np.min(complete_values)
        comp_cov = self.cov(self.amp2, self.ls, complete_configs)
        cand_cross = self.cov(self.amp2, self.ls, complete_configs, candidate_configs)

        obsv_cov = comp_cov + self.noise * np.eye(complete_configs.shape[0])
        obsv_chol = spla.cholesky(obsv_cov, lower=True)
        alpha = spla.cho_solve((obsv_chol, True), complete_values - self.mean)
        beta = spla.solve_triangular(obsv_chol, cand_cross, lower=True)

        func_m = np.dot(cand_cross.T, alpha) + self.mean
        func_v = self.amp2 * (1 + 1e-6) - np.sum(beta ** 2, axis=0)

        func_s = np.sqrt(func_v)
        u = (best - func_m + 0.01) / func_s
        n_cdf = sps.norm.cdf(u)
        n_pdf = sps.norm.pdf(u)
        ei = func_s * (u * n_cdf + n_pdf)

        if np.all(complete_costs <= self.budget):
            return ei
        else:
            return ei * weight

    def sample_constraint_hyper_params(self, complete_configs, complete_costs):
        if self.constraint_noiseless:
            self.constraint_noise = 1e-3
            self._sample_constraint_noiseless(complete_configs, complete_costs)
        else:
            self._sample_constraint_noisy(complete_configs, complete_costs)

        self._sample_constraint_ls(complete_configs, complete_costs)
        self.constraint_hyper_samples.append((self.constraint_mean, self.constraint_noise,
                                              self.constraint_amp2, self.constraint_ls))

    def sample_hyper_params(self, comp, values):
        if self.noiseless:
            self.noise = 1e-3
            self._sample_noiseless(comp, values)
        else:
            self._sample_noisy(comp, values)
        self._sample_ls(comp, values)

        self.hyper_samples.append((self.mean, self.noise, self.amp2, self.ls))

    def _sample_ls(self, comp, vals):
        def log_prob(ls):
            if np.any(ls < 0) or np.any(ls > self.max_ls):
                return -np.inf

            cov = (self.amp2 * (self.cov_func(ls, comp, None) + 1e-6 * np.eye(comp.shape[0]))
                   + self.noise * np.eye(comp.shape[0]))
            chol = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), vals - self.mean)
            lp = (-np.sum(np.log(np.diag(chol))) - 0.5 * np.dot(vals-self.mean, solve))
            return lp

        self.ls = util.slice_sample(self.ls, log_prob, compwise=True)

    def _sample_constraint_ls(self, comp, cos):
        def log_prob(ls):
            if np.any(ls < 0) or np.any(ls > self.constraint_max_ls):
                return -np.inf

            cov = (self.constraint_amp2 * (self.cov_func(ls, comp, None) + 1e-6 * np.eye(comp.shape[0])) +
                   self.constraint_noise * np.eye(comp.shape[0]))
            chol = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), cos - self.constraint_mean)
            lp = (-np.sum(np.log(np.diag(chol))) - 0.5 * np.dot(cos - self.constraint_mean, solve))
            return lp

        self.constraint_ls = util.slice_sample(self.constraint_ls, log_prob, compwise=True)

    def _sample_noisy(self, comp, values):
        def log_prob(hyper_params):
            mean = hyper_params[0]
            amp2 = hyper_params[1]
            noise = hyper_params[2]

            if mean > np.max(values) or mean < np.min(values):
                return -np.inf

            if amp2 < 0 or noise < 0:
                return -np.inf

            cov = amp2 * ((self.cov_func(self.ls, comp, None) + 1e-6 * np.eye(comp.shape[0])) +
                          noise * np.eye(comp.shape[0]))
            chol = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), values - mean)
            lp = -np.sum(np.log(np.diag(chol)))-0.5 * np.dot(values - mean, solve)

            lp += np.log(np.log(1 + (self.noise_scale / noise) ** 2))
            lp -= 0.5 * (np.log(amp2) / self.amp2_scale) ** 2
            return lp

        hyper_parameters = util.slice_sample(np.array([self.mean, self.amp2, self.noise]), log_prob, compwise=False)
        self.mean = hyper_parameters[0]
        self.amp2 = hyper_parameters[1]
        self.noise = hyper_parameters[2]

    def _sample_constraint_noisy(self, comp, cos):
        def log_prob(hypers):
            mean = hypers[0]
            amp2 = hypers[1]
            noise = hypers[2]

            if mean > np.max(cos) or mean < np.min(cos):
                return -np.inf

            if amp2 < 0 or noise < 0:
                return -np.inf

            cov = amp2 * ((self.cov_func(self.constraint_ls, comp, None) +
                            1e-6 * np.eye(comp.shape[0])) + noise * np.eye(comp.shape[0]))
            chol = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), cos - mean)
            lp = -np.sum(np.log(np.diag(chol))) - 0.5 * np.dot(cos - mean, solve)

            lp += np.log(np.log(1 + (self.constraint_noise_scale/noise)**2))
            lp -= 0.5 * (np.log(amp2) / self.constraint_amp2_scale)**2
            return lp

        hypers = util.slice_sample(np.array([self.constraint_mean, self.constraint_amp2, self.constraint_noise]),
                                   log_prob, compwise=False)
        self.constraint_mean = hypers[0]
        self.constraint_amp2 = hypers[1]
        self.constraint_noise = hypers[2]

    def _sample_noiseless(self, comp, vals):
        def log_prob(hypers):
            mean = hypers[0]
            amp2 = hypers[1]
            noise = 1e-3

            if mean > np.max(vals) or mean < np.min(vals):
                return -np.inf

            if amp2 < 0:
                return -np.inf

            cov = amp2 * ((self.cov_func(self.ls, comp, None) + 1e-6 * np.eye(comp.shape[0])) +
                          noise * np.eye(comp.shape[0]))
            chol = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), vals - mean)
            lp = -np.sum(np.log(np.diag(chol)))-0.5*np.dot(vals-mean, solve)

            lp -= 0.5*(np.log(amp2)/self.amp2_scale)**2

            return lp

        hypers = util.slice_sample(np.array(
                 [self.mean, self.amp2, self.noise]), log_prob, compwise=False)
        self.mean = hypers[0]
        self.amp2 = hypers[1]
        self.noise = 1e-3

    def _sample_constraint_noiseless(self, comp, cos):
        def log_prob(hypers):
            mean = hypers[0]
            amp2 = hypers[1]
            noise = 1e-3

            if mean > np.max(cos) or mean < np.min(cos):
                return -np.inf

            if amp2 < 0:
                return -np.inf

            cov = amp2 * ((self.cov_func(self.constraint_ls, comp, None) +
                           1e-6 * np.eye(comp.shape[0])) + noise * np.eye(comp.shape[0]))
            chol = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), cos - mean)
            lp = -np.sum(np.log(np.diag(chol))) - 0.5 * np.dot(cos - mean, solve)

            lp -= 0.5 * (np.log(amp2) / self.constraint_amp2_scale) ** 2

            return lp

        hypers = util.slice_sample(np.array([self.constraint_mean, self.constraint_amp2, self.constraint_noise]),
                                   log_prob, compwise=False)
        self.constraint_mean = hypers[0]
        self.constraint_amp2 = hypers[1]
        self.constraint_noise = 1e-3
